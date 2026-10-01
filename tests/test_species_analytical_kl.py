"""
Species analytical KL: the closed-form categorical KL at species noise eta = 0 and its
gradient flow in _compute_analytical_kl, plus the eta = 0 checks in the constructor.
"""
import math
import types

import pytest
import torch


def _make_bare_grpo(device="cpu", BK=4, fields=("pos", "cell", "species")):
    """Build the smallest possible GRPO instance to exercise
    _compute_analytical_kl in isolation. Stubs `self.si._data_fields`,
    `self.si._stochastic_interpolants`, and the pos `_epsilon` accessor so
    the field2si lookup works on every branch.
    """
    from omg.grpo.grpo_lightning import OMatGRPOModule

    g = OMatGRPOModule.__new__(OMatGRPOModule)
    g.fields = fields
    g.beta_kl_pos = 0.01
    g.beta_kl_species = 1.0
    g.beta_kl = 0.01
    # Fake si with the η=0 species SI exposed via the iteration protocol.
    # Pos branch needs _epsilon.epsilon(t); cell branch is dead-coded
    # (zero stub), so cell_si stays bare.
    species_si = types.SimpleNamespace(_noise=0.0)
    pos_si = types.SimpleNamespace(
        _epsilon=types.SimpleNamespace(
            epsilon=lambda t: torch.tensor(0.5, dtype=torch.float32),
        )
    )
    cell_si = types.SimpleNamespace()
    si = types.SimpleNamespace(
        _data_fields=[
            types.SimpleNamespace(name="pos"),
            types.SimpleNamespace(name="cell"),
            types.SimpleNamespace(name="species"),
        ],
        _stochastic_interpolants=[pos_si, cell_si, species_si],
    )
    g.si = si
    # Lightning's `self.device` is a property descriptor reading `_device`
    # off `_DeviceDtypeModuleMixin`. Setting g.__dict__["device"] does NOT
    # shadow it; assigning `_device` does.
    g._device = torch.device(device)
    # Sanity: confirm the property routes through.
    assert g.device == torch.device(device), (
        f"Lightning .device property override failed: got {g.device}"
    )
    return g


def _make_traj(T, BK, device="cpu"):
    times = torch.linspace(0.05, 0.95, T + 1, device=device)[:-1]
    dt = torch.full((T,), (0.95 - 0.05) / T, device=device)
    return {
        "times": times,
        "dt": dt,
        "batch": torch.arange(BK, device=device),  # one atom per struct
    }


def _zeros_drifts(BK, T, device="cpu"):
    return {
        "pos": torch.zeros(T, BK, 3, device=device),
        "cell": torch.zeros(T, BK, 3, 3, device=device),
    }


# ------------- math correctness -------------


def test_compute_analytical_kl_returns_dict():
    g = _make_bare_grpo()
    BK, T = 2, 4
    traj = _make_traj(T, BK)
    n_atoms = torch.ones(BK)
    drifts = _zeros_drifts(BK, T)
    kl = g._compute_analytical_kl(drifts, drifts, traj, n_atoms, BK,
                                  species_dist_new=None, species_dist_ref=None)
    # No KL term on the lattice channel: the dict has no "cell" entry.
    assert set(kl.keys()) == {"pos", "species"}
    for k, v in kl.items():
        assert isinstance(v, torch.Tensor)
        assert v.dim() == 0


def test_species_kl_zero_when_models_match():
    """π_θ ≡ π_θref ⇒ KL_species = 0 (within fp tolerance)."""
    g = _make_bare_grpo()
    BK, T = 2, 4
    S = 118
    traj = _make_traj(T, BK)
    n_atoms = torch.ones(BK)
    E = 6
    same_logits = torch.randn(E, S)
    species_dist = {
        "logits": same_logits.clone(),
        "step_idx": torch.zeros(E, dtype=torch.long),
        "struct_idx": torch.zeros(E, dtype=torch.long),
        "p_unmask": torch.ones(E),
    }
    species_dist_ref = {k: v.clone() for k, v in species_dist.items()}
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=species_dist,
        species_dist_ref=species_dist_ref,
    )
    assert kl["species"].abs().item() < 1e-6


def test_species_kl_one_hot_vs_uniform():
    """KL[Cat(one-hot at z*) || Cat(Uniform)] = log(S) per event.

    Single masked atom in single struct, p_unmask=1, T=1, n_atoms=1
    ⇒ kl["species"] = log(S) / T = log(S).
    """
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    big = 1e4
    z_star = 17
    new_logits = torch.full((1, 118), -big)
    new_logits[0, z_star] = big      # one-hot at z*
    ref_logits = torch.zeros(1, 118)  # uniform
    species_dist_new = {
        "logits": new_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    species_dist_ref = {
        "logits": ref_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    S = species_dist_new["logits"].shape[-1]
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=species_dist_new,
        species_dist_ref=species_dist_ref,
    )
    # KL_total = (kl_per_struct / n_atoms).mean() / T = log(S) / 1.
    expected = math.log(S)
    assert abs(kl["species"].item() - expected) < 1e-3, (
        f"got {kl['species'].item()}, expected {expected}"
    )


def test_species_kl_uniform_vs_uniform():
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    species_dist = {
        "logits": torch.zeros(1, 118),
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=species_dist,
        species_dist_ref={k: v.clone() for k, v in species_dist.items()},
    )
    assert kl["species"].abs().item() < 1e-7


def test_species_kl_p_unmask_linearity():
    """KL scales linearly in p_unmask (the schedule term is multiplicative)."""
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    new_logits = torch.zeros(1, 118)
    new_logits[0, 0] = 5.0
    ref_logits = torch.zeros(1, 118)
    base = {
        "logits": new_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
    }
    ref = {
        "logits": ref_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
    }
    kls = []
    for pu in [0.1, 0.5, 1.0]:
        new_d = {**base, "p_unmask": torch.tensor([pu])}
        ref_d = {**ref, "p_unmask": torch.tensor([pu])}
        kl = g._compute_analytical_kl(
            _zeros_drifts(BK, T), _zeros_drifts(BK, T),
            traj, n_atoms, BK,
            species_dist_new=new_d, species_dist_ref=ref_d,
        )
        kls.append(kl["species"].item())
    # Ratios should match: kl(0.5)/kl(0.1) ≈ 5, kl(1.0)/kl(0.1) ≈ 10.
    assert abs(kls[1] / kls[0] - 5.0) < 1e-3
    assert abs(kls[2] / kls[0] - 10.0) < 1e-3


def test_species_kl_eta_assert():
    """Hard guard: noise != 0 must raise."""
    g = _make_bare_grpo()
    g.si._stochastic_interpolants[2]._noise = 0.189  # bad
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    species_dist = {
        "logits": torch.zeros(1, 118),
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    with pytest.raises(AssertionError, match="η=0|noise"):
        g._compute_analytical_kl(
            _zeros_drifts(BK, T), _zeros_drifts(BK, T),
            traj, n_atoms, BK,
            species_dist_new=species_dist,
            species_dist_ref={k: v.clone() for k, v in species_dist.items()},
        )


def test_species_kl_alignment_assert():
    """Mismatched step_idx between new/ref must raise."""
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    new = {
        "logits": torch.zeros(1, 118),
        "step_idx": torch.tensor([0], dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    ref = {**new, "step_idx": torch.tensor([1], dtype=torch.long)}
    with pytest.raises(AssertionError, match="step_idx"):
        g._compute_analytical_kl(
            _zeros_drifts(BK, T), _zeros_drifts(BK, T),
            traj, n_atoms, BK,
            species_dist_new=new, species_dist_ref=ref,
        )


def test_species_kl_empty_event_no_crash():
    g = _make_bare_grpo()
    BK, T = 2, 4
    traj = _make_traj(T, BK)
    n_atoms = torch.ones(BK)
    empty = {
        "logits": torch.zeros(0, 118),
        "step_idx": torch.zeros(0, dtype=torch.long),
        "struct_idx": torch.zeros(0, dtype=torch.long),
        "p_unmask": torch.zeros(0),
    }
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=empty,
        species_dist_ref={k: v.clone() for k, v in empty.items()},
    )
    assert kl["species"].abs().item() == 0.0


# ------------- gradient flow -------------


def test_species_kl_grad_flow_to_new_logits():
    """Gradient of the species KL must flow back through new_logits."""
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    new_logits = torch.randn(1, 118, requires_grad=True)
    ref_logits = torch.randn(1, 118)
    new_d = {
        "logits": new_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    ref_d = {
        "logits": ref_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=new_d, species_dist_ref=ref_d,
    )
    kl["species"].backward()
    assert new_logits.grad is not None
    assert new_logits.grad.norm().item() > 0.0


def test_diagnostic_kl_total_has_grad():
    """Pin the live-tensor contract for the diagnostic backward.

    stats["loss/kl"] is detached for logging, so the diagnostic backward in
    training_step must rebuild kl_total from analytical_kl[*]. A regression
    that reads from stats[...] would silently zero grad/kl_only_norm
    forever. This test fails hard if the live reconstruction is broken
    (e.g. someone .detach()s analytical_kl entries).
    """
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    new_logits = torch.randn(1, 118, requires_grad=True)
    ref_logits = torch.randn(1, 118)
    new_d = {
        "logits": new_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    ref_d = {
        "logits": ref_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    analytical_kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=new_d, species_dist_ref=ref_d,
    )
    # Mirror the diagnostic-site reconstruction in training_step.
    kl_total_t = (
        g.beta_kl_pos * analytical_kl["pos"]
        + g.beta_kl_species * analytical_kl["species"]
    )
    assert kl_total_t.requires_grad, (
        "Diagnostic kl_total_t lost its grad — detach somewhere upstream. "
        "The training_step diagnostic backward will become a silent no-op. "
        "It must be rebuilt from analytical_kl, not read from stats."
    )
    kl_total_t.backward()
    assert new_logits.grad is not None
    assert new_logits.grad.norm().item() > 0.0


def test_species_kl_no_grad_through_ref():
    """ref_logits must not be a parameter of the autograd graph."""
    g = _make_bare_grpo()
    BK, T = 1, 1
    traj = {
        "times": torch.tensor([0.5]),
        "dt": torch.tensor([0.1]),
        "batch": torch.tensor([0]),
    }
    n_atoms = torch.ones(BK)
    new_logits = torch.randn(1, 118, requires_grad=True)
    ref_logits = torch.randn(1, 118, requires_grad=False)
    new_d = {
        "logits": new_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    ref_d = {
        "logits": ref_logits,
        "step_idx": torch.zeros(1, dtype=torch.long),
        "struct_idx": torch.zeros(1, dtype=torch.long),
        "p_unmask": torch.ones(1),
    }
    kl = g._compute_analytical_kl(
        _zeros_drifts(BK, T), _zeros_drifts(BK, T),
        traj, n_atoms, BK,
        species_dist_new=new_d, species_dist_ref=ref_d,
    )
    kl["species"].backward()
    assert ref_logits.grad is None


# ------------- analytical_kl argument type -------------


def test_legacy_caller_compat_dict():
    """_ppo_grpo_loss accepts the new dict-typed analytical_kl."""
    g = _make_bare_grpo()
    g.eps_clip = 0.2
    g.alpha_pos = 1.0
    g.alpha_cell = 1.0
    g.alpha_species = 1.0
    BK = 2
    T = 4
    advantages = torch.tensor([1.0, -1.0])
    logp_old = {
        "pos": torch.zeros(BK, T),
        "cell": torch.zeros(BK, T),
    }
    logp_new = {
        "pos": torch.zeros(BK, T, requires_grad=True),
        "cell": torch.zeros(BK, T, requires_grad=True),
    }
    g.fields = ("pos", "cell")
    analytical_kl = {"pos": torch.tensor(0.05), "cell": torch.tensor(0.0),
                     "species": torch.tensor(0.0)}
    loss, stats = g._ppo_grpo_loss(
        logp_old, logp_new, advantages,
        analytical_kl=analytical_kl,
    )
    expected = g.beta_kl_pos * 0.05
    assert abs(stats["loss/kl"].item() - expected) < 1e-7


def test_legacy_caller_rejects_scalar_tensor():
    """Passing a scalar tensor instead of the per-channel dict raises a clear TypeError."""
    g = _make_bare_grpo()
    g.eps_clip = 0.2
    g.alpha_pos = 1.0
    g.alpha_cell = 1.0
    g.alpha_species = 1.0
    g.fields = ("pos", "cell")
    BK = 2
    T = 4
    advantages = torch.tensor([1.0, -1.0])
    logp_old = {"pos": torch.zeros(BK, T), "cell": torch.zeros(BK, T)}
    logp_new = {
        "pos": torch.zeros(BK, T, requires_grad=True),
        "cell": torch.zeros(BK, T, requires_grad=True),
    }
    with pytest.raises(TypeError, match="Dict"):
        g._ppo_grpo_loss(
            logp_old, logp_new, advantages,
            analytical_kl=torch.tensor(0.05),  # scalar instead of dict
        )


# ------------- η override + early init guard -------------


def _make_si_with_species_eta(eta):
    """Build the minimal `si` object the early init guard inspects.
    `_data_fields` mirrors the production order pos/cell/species; the species
    SI exposes `_noise = eta`."""
    species_si = types.SimpleNamespace(_noise=eta)
    pos_si = types.SimpleNamespace(
        _epsilon=types.SimpleNamespace(epsilon=lambda t: torch.tensor(0.5))
    )
    cell_si = types.SimpleNamespace()
    return types.SimpleNamespace(
        _data_fields=[
            types.SimpleNamespace(name="pos"),
            types.SimpleNamespace(name="cell"),
            types.SimpleNamespace(name="species"),
        ],
        _stochastic_interpolants=[pos_si, cell_si, species_si],
    )


class _StubModel(torch.nn.Module):
    """Tiny nn.Module so copy.deepcopy(model).eval() in __init__ works without
    pulling in the real CSPNet."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)


def _patched_init(*args, **kwargs):
    """Mock `OMatGRPOReward` in __init__ so unit tests don't load
    UMA (~10 s + GPU + network on first call).
    Returns a no-op SimpleNamespace stand-in for the reward."""
    from unittest.mock import patch
    return patch(
        "omg.grpo.grpo_lightning.OMatGRPOReward",
        return_value=types.SimpleNamespace(),
    )


def test_init_guard_accepts_eta_zero():
    """GRPO __init__ must not raise when species ∈ fields and species SI
    has noise=0.0 and beta_kl_species > 0."""
    from omg.grpo.grpo_lightning import OMatGRPOModule

    si = _make_si_with_species_eta(0.0)
    with _patched_init():
        g = OMatGRPOModule(
            si=si, sampler=types.SimpleNamespace(), model=_StubModel(),
            k=4, fields=("pos", "cell", "species"),
            beta_kl=0.01, beta_kl_species=1e-3,
        )
    assert g.beta_kl_species == 1e-3


def test_init_guard_rejects_eta_nonzero():
    """GRPO __init__ must raise when species ∈ fields and species SI
    noise > 0 and beta_kl_species > 0 (prior configs carry eta > 0)."""
    from omg.grpo.grpo_lightning import OMatGRPOModule

    si = _make_si_with_species_eta(0.18946955217679085)
    # Guard fires BEFORE reward init, so no need to patch reward.
    with pytest.raises(RuntimeError, match="species_eta_override"):
        OMatGRPOModule(
            si=si, sampler=types.SimpleNamespace(), model=_StubModel(),
            k=4, fields=("pos", "cell", "species"),
            beta_kl=0.01, beta_kl_species=1e-3,
        )


def test_init_guard_skipped_when_species_off():
    """Init guard must NOT fire when species ∉ fields (CSP-only) even
    if species SI has noise > 0 — the species path is inert."""
    from omg.grpo.grpo_lightning import OMatGRPOModule

    si = _make_si_with_species_eta(0.189)
    with _patched_init():
        g = OMatGRPOModule(
            si=si, sampler=types.SimpleNamespace(), model=_StubModel(),
            k=4, fields=("pos", "cell"),    # species NOT in fields
            beta_kl=0.01, beta_kl_species=1e-3,
        )
    assert g.fields == ("pos", "cell")


def test_init_guard_skipped_when_beta_kl_species_zero():
    """Init guard must NOT fire when beta_kl_species == 0.0 even if
    species ∈ fields — species KL channel is inert."""
    from omg.grpo.grpo_lightning import OMatGRPOModule

    si = _make_si_with_species_eta(0.189)
    with _patched_init():
        g = OMatGRPOModule(
            si=si, sampler=types.SimpleNamespace(), model=_StubModel(),
            k=4, fields=("pos", "cell", "species"),
            beta_kl=0.01, beta_kl_species=0.0,
        )
    assert g.beta_kl_species == 0.0
