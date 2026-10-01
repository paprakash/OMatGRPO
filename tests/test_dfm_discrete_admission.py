"""
The discrete species channel (DifferentialEquationType.DISCRETE) is admitted as a
learnable field alongside the SDE channels at every place that checks field types.

Cases:
  1. test_dfm_marks_itself_discrete - the species interpolant reports DISCRETE.
  2. test_cli_admits_species_via_helper - the entry point's field admission accepts species.
  3. test_integrate_and_step_logprob_admit_discrete - rollout and replay accept species.
"""
import pytest
import torch
from torch_geometric.data import Data

from omg.globals import MAX_ATOM_NUM
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from omg.si.single_stochastic_interpolant import (
    DifferentialEquationType,
    SingleStochasticInterpolant,
)
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.si.gamma import LatentGammaSqrt
from omg.si.epsilon import VanishingEpsilon
from omg.grpo.train import _admit_fields


# =====================================================================
# Case 1 — DFM sentinel attribute
# =====================================================================

def test_dfm_marks_itself_discrete():
    si = DiscreteFlowMatchingMask(noise=0.0)
    assert si._differential_equation_type is DifferentialEquationType.DISCRETE


# =====================================================================
# Case 2 — _admit_fields helper (Site 2, unit test)
# =====================================================================

class _FakeSI:
    def __init__(self, de_type):
        self._differential_equation_type = de_type


def _mock_field2si():
    return {
        "pos":     _FakeSI(DifferentialEquationType.SDE),
        "cell":    _FakeSI(DifferentialEquationType.SDE),
        "species": _FakeSI(DifferentialEquationType.DISCRETE),
    }


def test_admit_fields_all_admissible():
    """All requested fields admissible: returned unchanged."""
    assert _admit_fields(_mock_field2si(), ("pos", "cell", "species")) == ("pos", "cell", "species")


def test_admit_fields_unknown_field_errors():
    """A requested field without an SDE/DISCRETE interpolant raises (no silent downgrade)."""
    with pytest.raises(RuntimeError, match="no stochastic"):
        _admit_fields(_mock_field2si(), ("pos", "xyz"))


def test_admit_fields_rejects_ode_type():
    """ODE fields are not admissible (SDE and DISCRETE only)."""
    f2si = {
        "pos":  _FakeSI(DifferentialEquationType.SDE),
        "cell": _FakeSI(DifferentialEquationType.ODE),
    }
    with pytest.raises(RuntimeError, match="Learnable fields of this model: \\['pos'\\]"):
        _admit_fields(f2si, ("pos", "cell"))


# =====================================================================
# Case 3 — integrate_with_logprob + step_logprob admit DISCRETE (Sites 3+4)
# =====================================================================

class _ToyModelWithSpecies(torch.nn.Module):
    """ToyModel variant that additionally emits a species_b head of shape
    (sum_atoms, MAX_ATOM_NUM), as DFM.integrate consumes it via
    model_prediction_fn_species -> preds_ref['species_b']."""

    def __init__(self) -> None:
        super().__init__()
        self.species_embed = torch.nn.Embedding(120, 8)
        self.pos_mlp = torch.nn.Linear(3 + 8 + 1, 6)
        self.cell_mlp = torch.nn.Linear(9 + 1, 18)
        self.species_mlp = torch.nn.Linear(3 + 8 + 1, MAX_ATOM_NUM)

    def forward(self, x, t):
        t_per_atom = t[x.batch].unsqueeze(-1)
        sp_emb = self.species_embed(x.species.long())
        pos_in = torch.cat([x.pos, sp_emb, t_per_atom], dim=-1)
        pos_out = self.pos_mlp(pos_in)
        pos_b, pos_eta = pos_out[:, :3], pos_out[:, 3:]

        B = x.cell.shape[0]
        cell_flat = x.cell.reshape(B, 9)
        cell_in = torch.cat([cell_flat, t.unsqueeze(-1)], dim=-1)
        cell_out = self.cell_mlp(cell_in)
        cell_b = cell_out[:, :9].reshape(B, 3, 3)
        cell_eta = cell_out[:, 9:].reshape(B, 3, 3)

        species_logits = self.species_mlp(pos_in)  # (sum_atoms, MAX_ATOM_NUM)

        return Data(
            pos_b=pos_b, pos_eta=pos_eta,
            cell_b=cell_b, cell_eta=cell_eta,
            species_b=species_logits,
        )


def _build_x0_masked(seed: int = 0) -> Data:
    """x0 with ALL species masked (token 0). DFM requires mask-start."""
    torch.manual_seed(seed)
    B = 2
    n_per = [4, 4]
    n_atoms = torch.tensor(n_per, dtype=torch.long)
    total_atoms = int(n_atoms.sum())
    batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long), n_atoms)
    ptr = torch.zeros(B + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(n_atoms, dim=0)
    pos = torch.rand(total_atoms, 3)
    cell = torch.randn(B, 3, 3) * 0.3 + torch.eye(3).unsqueeze(0) * 5.0
    species = torch.zeros(total_atoms, dtype=torch.long)  # all masked
    return Data(pos=pos, cell=cell, species=species,
                batch=batch, n_atoms=n_atoms, ptr=ptr)


def _build_si_with_species(T: int = 8) -> StochasticInterpolants:
    """pos=SDE + cell=SDE + species=DFM(DISCRETE)."""
    pos_si = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(),
        gamma=LatentGammaSqrt(0.1),
        epsilon=VanishingEpsilon(c=0.1),
        differential_equation_type='SDE',
        integrator_kwargs={'method': 'srk'},
    )
    cell_si = SingleStochasticInterpolant(
        interpolant=LinearInterpolant(),
        gamma=LatentGammaSqrt(0.1),
        epsilon=VanishingEpsilon(c=0.1),
        differential_equation_type='SDE',
        integrator_kwargs={'method': 'srk'},
    )
    species_si = DiscreteFlowMatchingMask(noise=0.0)
    return StochasticInterpolants(
        stochastic_interpolants=[pos_si, cell_si, species_si],
        data_fields=['pos', 'cell', 'species'],
        integration_time_steps=T,
        enable_progress_bar=False,
    )


@pytest.mark.parametrize(
    "fields",
    [("pos", "cell", "species"), ("species",)],
    ids=["fields=pos_cell_species", "fields=species_only"],
)
def test_integrate_and_step_logprob_admit_discrete(fields):
    """Sites 3+4: assertion blocks must admit DISCRETE. Execution must complete
    without the old `de_type == SDE` assertion firing, and returned tensors
    must have the documented shapes. This is the admission shield, not a PPO
    ratio check (that's Step 1b stretch)."""
    torch.manual_seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    x0 = _build_x0_masked(seed=0)
    model = _ToyModelWithSpecies()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    si = _build_si_with_species(T=8)

    torch.manual_seed(42)
    gen, traj, logp_old_step = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=fields, stochastic=True, return_trajectory=True,
    )

    B = 2
    sum_atoms = 8
    # integrate_with_logprob returns a per-field dict. Pos/cell
    # (when present in fields) carry [B, T_inner]; T_inner = len(times) - 1, so
    # with integration_time_steps=8, inner T is 7. Derive from whichever field
    # is in the dict.
    assert isinstance(logp_old_step, dict)
    _primary = "pos" if "pos" in logp_old_step else ("cell" if "cell" in logp_old_step else None)
    if _primary is not None:
        assert logp_old_step[_primary].shape[0] == B
        T_inner = logp_old_step[_primary].shape[1]
    else:
        # species-only run: no per-step tensors to derive T from. T_inner is
        # the species_seq first dim.
        assert "species_seq" in traj
        T_inner = traj["species_seq"].shape[0]
    assert "species_seq" in traj
    assert traj["species_seq"].shape == (T_inner, sum_atoms)
    assert "pos_next_seq" in traj
    assert "cell_next_seq" in traj

    logp_new_step = si.step_logprob(model, traj, fields=fields)
    assert isinstance(logp_new_step, dict)
    # Per-field shape match for pos/cell when present
    for fname in ("pos", "cell"):
        if fname in logp_old_step:
            assert logp_new_step[fname].shape == logp_old_step[fname].shape
            assert torch.isfinite(logp_new_step[fname]).all()
    # Species event shape — all-M start → species evolves → "species" key present.
    if "species" in logp_old_step:
        assert "species" in logp_new_step
        sp_old = logp_old_step["species"]
        sp_new = logp_new_step["species"]
        assert sp_new["logp"].shape == sp_old["logp"].shape
        assert torch.isfinite(sp_new["logp"]).all()
