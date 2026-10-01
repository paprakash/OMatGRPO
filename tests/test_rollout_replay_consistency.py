"""
Epoch-0 PPO ratio invariant: rollout logp must equal replay logp.

Locks in the fix from commit c8e2d92 on branch fix/integrate-all-fields:
  - integrate_with_logprob always evolves pos AND cell every step (on-manifold
    joint (pos,cell,species) states) and always stores pos_next_seq /
    cell_next_seq in traj, regardless of which field is in `fields`.
  - step_logprob always advances pos_curr and cell_curr from the stored
    trajectory, regardless of which field is in `fields`.

Without this: if fields=("pos",), cell froze at t=0 noise during rollout and
stayed at cell_init during replay, which made rollout off-manifold AND made
logp_new != logp_old at epoch 0 — corrupting the PPO ratio before any
learning step.

The invariant tested here (logp_old_step == logp_new_step at epoch 0, with
the same unchanged model) is weight-independent; the toy model is only
required to be deterministic given its inputs.
"""
import pytest
import torch
from torch_geometric.data import Data

from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.si.gamma import LatentGammaSqrt
from omg.si.epsilon import VanishingEpsilon


class ToyModel(torch.nn.Module):
    """
    Minimal model exposing the heads consumed by integrate_with_logprob /
    step_logprob: pos_b, pos_eta (per-atom, 3-d), cell_b, cell_eta (per-
    structure, 3x3). Pure-function given inputs — no dropout, no
    batchnorm — so deterministic evaluation gives bitwise-identical preds
    when called twice on the same state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.species_embed = torch.nn.Embedding(120, 8)
        self.pos_mlp = torch.nn.Linear(3 + 8 + 1, 6)   # -> pos_b (3) + pos_eta (3)
        self.cell_mlp = torch.nn.Linear(9 + 1, 18)     # -> cell_b (9) + cell_eta (9)

    def forward(self, x, t):
        # t is shape (B,): broadcast to per-atom via x.batch for the pos head.
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

        return Data(pos_b=pos_b, pos_eta=pos_eta,
                    cell_b=cell_b, cell_eta=cell_eta)


def _build_x0(seed: int = 0) -> Data:
    """Tiny Data batch: B=2 structures, 4 atoms each, random pos/cell/species."""
    torch.manual_seed(seed)
    B = 2
    n_per = [4, 4]
    n_atoms = torch.tensor(n_per, dtype=torch.long)
    total_atoms = int(n_atoms.sum())
    batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long), n_atoms)
    ptr = torch.zeros(B + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(n_atoms, dim=0)
    pos = torch.rand(total_atoms, 3)                                  # fractional
    cell = torch.randn(B, 3, 3) * 0.3 + torch.eye(3).unsqueeze(0) * 5.0
    species = torch.randint(1, 50, (total_atoms,), dtype=torch.long)
    return Data(pos=pos, cell=cell, species=species,
                batch=batch, n_atoms=n_atoms, ptr=ptr)


def _build_si(T: int = 8) -> StochasticInterpolants:
    """
    Pos + cell SIs only, both SDE. Matches the minimum requirements of
    integrate_with_logprob (species is optional and not exercised here —
    the species gate stays inert when 'species' not in fields and species
    is not in field2si).
    """
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
    return StochasticInterpolants(
        stochastic_interpolants=[pos_si, cell_si],
        data_fields=['pos', 'cell'],
        integration_time_steps=T,
        enable_progress_bar=False,
    )


@pytest.mark.parametrize("fields", [("pos", "cell"), ("pos",)],
                         ids=["fields=pos_cell", "fields=pos_only"])
def test_epoch0_logprob_invariance(fields):
    """
    After rollout, re-running step_logprob under the *same* model must
    reproduce the rollout's per-step per-structure log-prob to within
    float32 numerical noise. Breaking this means the PPO ratio at inner
    epoch 0 is not ~1.0 and advantages are applied to garbage.
    """
    torch.manual_seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    x0 = _build_x0(seed=0)
    model = ToyModel()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    si = _build_si(T=8)

    torch.manual_seed(42)
    gen, traj, logp_old_step = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=fields, stochastic=True, return_trajectory=True,
    )

    # Fix c8e2d92: both buffers must be allocated unconditionally so step_logprob
    # can replay the same evolved state sequence whether or not a field is learned.
    assert "pos_next_seq" in traj, (
        f"traj missing 'pos_next_seq' for fields={fields}. "
        "integrate_with_logprob must allocate pos_next_seq unconditionally "
        "(fix c8e2d92 in omg/si/stochastic_interpolants.py)."
    )
    assert "cell_next_seq" in traj, (
        f"traj missing 'cell_next_seq' for fields={fields}. "
        "integrate_with_logprob must allocate cell_next_seq unconditionally "
        "(fix c8e2d92 in omg/si/stochastic_interpolants.py)."
    )

    # Last trajectory row must equal the returned final state — a direct check
    # that every step writes through regardless of fields membership.
    assert torch.allclose(traj["pos_next_seq"][-1], gen.pos, atol=1e-6, rtol=0), (
        f"traj['pos_next_seq'][-1] != gen.pos for fields={fields}; "
        "pos state storage is out of sync with pos state evolution."
    )
    assert torch.allclose(traj["cell_next_seq"][-1], gen.cell, atol=1e-6, rtol=0), (
        f"traj['cell_next_seq'][-1] != gen.cell for fields={fields}; "
        "cell state storage is out of sync with cell state evolution "
        "(likely the 'cell' in fields gate on storage has been re-introduced)."
    )

    # Replay under the same (unchanged) model — should reproduce logp_old_step.
    logp_new_step = si.step_logprob(model, traj, fields=fields)

    # Rollout and replay both return a per-field dict.
    # SI built without species → "species" key absent; only pos/cell assertions apply.
    assert isinstance(logp_old_step, dict) and isinstance(logp_new_step, dict)
    for fname in fields:
        if fname not in ("pos", "cell"):
            continue
        old_f = logp_old_step[fname]
        new_f = logp_new_step[fname]
        assert new_f.shape == old_f.shape, (
            f"shape mismatch logp_new[{fname}]={tuple(new_f.shape)} "
            f"vs logp_old[{fname}]={tuple(old_f.shape)} for fields={fields}"
        )

        diff = (new_f - old_f).abs()
        max_abs = diff.max().item()
        flat_idx = int(diff.argmax().item())
        T = old_f.shape[1]
        bad_struct, bad_step = divmod(flat_idx, T)
        assert torch.allclose(new_f, old_f, atol=1e-5, rtol=1e-4), (
            f"Epoch-0 PPO ratio invariant violated for fields={fields}, "
            f"channel={fname}:\n"
            f"  max |logp_new - logp_old| = {max_abs:.3e} "
            f"at (structure={bad_struct}, step={bad_step})\n"
            f"  logp_old[{bad_struct},{bad_step}] = {old_f[bad_struct, bad_step].item():.6f}\n"
            f"  logp_new[{bad_struct},{bad_step}] = {new_f[bad_struct, bad_step].item():.6f}\n"
            "This means the replay state sequence in step_logprob has diverged from\n"
            "the rollout state sequence in integrate_with_logprob. Verify that\n"
            "(a) pos_next_seq and cell_next_seq are allocated + written every step\n"
            "    regardless of `fields` in integrate_with_logprob, and\n"
            "(b) pos_curr = traj['pos_next_seq'][j] and\n"
            "    cell_curr = traj['cell_next_seq'][j] execute unconditionally\n"
            "    (outside the `if fname in fields:` blocks) in step_logprob.\n"
            "See commit c8e2d92 on branch fix/integrate-all-fields."
        )
