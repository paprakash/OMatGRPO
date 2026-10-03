"""The reward variants of the reward-hacking appendix, on CPU without UMA.

TorchSim's static and optimize calls and the hull lookup are replaced by fakes with known
outputs, so each variant's arithmetic can be checked exactly: the absolute-energy and
formation-energy rewards, the displacement reward and the residual geometry term. The command
line flags of the variants are checked against the reward configuration they produce.
"""
import math

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import omg.grpo.reward as reward_mod
from omg.grpo.reward import OMatGRPOReward
from omg.grpo.train import module_kwargs, resolve_config

E_PER_ATOM = -3.0      # energy per atom returned by the fake potential
SHIFT = 0.1            # displacement (A) applied to every atom by the fake relaxation


def _structures():
    s1 = Structure(Lattice.cubic(4.0), ["Cs", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return [s1, s2]


class _State:
    def __init__(self, atoms):
        self._atoms = atoms

    def to_atoms(self):
        return self._atoms


@pytest.fixture
def fake_torchsim(monkeypatch, no_uma):
    def static(system, model):
        return [{"potential_energy": torch.tensor(E_PER_ATOM * len(a))} for a in system]

    def optimize(system, model, **kwargs):
        moved = []
        for a in system:
            b = a.copy()
            b.positions = b.positions + np.array([SHIFT, 0.0, 0.0])
            moved.append(b)
        return _State(moved)

    monkeypatch.setattr(reward_mod.ts, "static", static)
    monkeypatch.setattr(reward_mod.ts, "optimize", optimize)


def test_absolute_energy_reward(fake_torchsim):
    r = OMatGRPOReward(device="cpu", reward_type="absolute")
    out = r.calculate_batch_energy_reward(_structures())
    assert torch.equal(out, torch.full((2,), -E_PER_ATOM))


def test_formation_energy_reward(fake_torchsim):
    r = OMatGRPOReward(device="cpu", reward_type="formation")
    out = r.calculate_batch_energy_reward(_structures())
    expected = -torch.clamp(torch.tensor(r.last_formation_energy_per_atom), -10.0, 10.0)
    assert all(math.isfinite(x) for x in r.last_formation_energy_per_atom)
    assert torch.allclose(out, expected)


def test_residual_geometry_term(fake_torchsim, monkeypatch):
    r = OMatGRPOReward(device="cpu", reward_type="e_hull", relax_before_reward=True,
                       w_rmsd_geom=1.0, rmsd_geom_clamp=3.0, ehull_sparse_gate=True,
                       sparse_route="neutral", ehull_floor_at_zero=True, ehull_cap=1.0)
    monkeypatch.setattr(r, "_lookup_e_hull", lambda atoms, ok_idx, e: ([0.05, 0.2], [20, 20]))
    out = r.calculate_batch_energy_reward(_structures())
    # stability term -E_hull, minus the RMSD between generated and relaxed structure (SHIFT)
    assert r.last_rmsd_geom_clamped == pytest.approx([SHIFT, SHIFT])
    assert out.tolist() == pytest.approx([-0.05 - SHIFT, -0.2 - SHIFT])


def test_residual_geometry_term_off_leaves_stability_term(fake_torchsim, monkeypatch):
    r = OMatGRPOReward(device="cpu", reward_type="e_hull", relax_before_reward=True,
                       ehull_sparse_gate=True, sparse_route="neutral", ehull_floor_at_zero=True,
                       ehull_cap=1.0)
    monkeypatch.setattr(r, "_lookup_e_hull", lambda atoms, ok_idx, e: ([0.05, 0.2], [20, 20]))
    out = r.calculate_batch_energy_reward(_structures())
    assert out.tolist() == pytest.approx([-0.05, -0.2])


def test_displacement_reward(fake_torchsim):
    r = OMatGRPOReward(device="cpu", weights={"rmsd": 2.0, "energy": 1.0}, reward_offset=0.3)
    out = r.calculate_batch_rmsd_reward(_structures(), "rmsd")
    expected = 2.0 * (math.log(1.3) - math.log(1.0 + SHIFT))
    assert out.tolist() == pytest.approx([expected, expected])


def test_fmax_schedule(no_uma):
    r = OMatGRPOReward(device="cpu", fmax=10.0, fmax_schedule=[(0, 10.0), (100, 5.0), (300, 1.0)])
    for step, fmax in [(0, 10.0), (99, 10.0), (100, 5.0), (299, 5.0), (300, 1.0), (10_000, 1.0)]:
        r._current_step = step
        assert r._get_current_fmax() == fmax


def test_unknown_reward_type_is_rejected(no_uma):
    with pytest.raises(ValueError, match="reward_type"):
        OMatGRPOReward(device="cpu", reward_type="bogus")


def test_variant_flags_reach_the_reward_config():
    cfg = resolve_config(["--reward_type", "formation", "--creat_on_relaxed", "false",
                          "--w_rmsd", "0.5", "--fmax", "5", "--fmax_schedule", "0:10,1000:5",
                          "--reward_offset", "0.2", "--w_rmsd_geom", "1", "--rmsd_geom_clamp", "2"],
                         env={"OMATGRPO_DATA_DIR": "/nonexistent"})
    kw = module_kwargs(cfg)
    rc = kw["reward_cfg"]
    assert rc["reward_type"] == "formation"
    assert rc["weights"] == {"rmsd": 0.5, "energy": 1.0}
    assert rc["fmax"] == 5.0 and rc["fmax_schedule"] == [(0, 10.0), (1000, 5.0)]
    assert rc["w_rmsd_geom"] == 1.0 and rc["rmsd_geom_clamp"] == 2.0
    assert kw["reward_offset"] == 0.2


def test_creativity_on_relaxed_needs_the_stability_reward():
    with pytest.raises(ValueError, match="reward_type"):
        resolve_config(["--reward_type", "absolute"], env={"OMATGRPO_DATA_DIR": "/nonexistent"})
