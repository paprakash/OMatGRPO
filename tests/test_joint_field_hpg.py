"""
Joint-field training checks (positions, lattice and species learned together).

  T1 - alpha_species multiplies the species term of
       OMatGRPOModule._ppo_grpo_loss.
  T3 - _apply_mask_guard flags arrays that contain a mask token (Z=0), which replaces
       them with Z=1 and routes the structure away from UMA, and passes fully resolved
       arrays through unchanged.
  T4 - the entry point passes --alpha_species through to OMatGRPOModule.
  T5 - step_logprob(return_drift=True) does not stack drift lists of channels without
       drift entries (the discrete species channel has none); torch.stack([]) would raise.
"""
import inspect
import re

import numpy as np
import pytest
import torch
import lightning as L

from omg.grpo.grpo_lightning import OMatGRPOModule


# =====================================================================
# T1 — α_species threads into _ppo_grpo_loss
# =====================================================================

def _make_lm_with_alpha_species(alpha_species: float) -> OMatGRPOModule:
    """Bare Lightning instance with only the knobs _ppo_grpo_loss touches.
    Avoids the full __init__ (which loads the UMA reward model)."""
    lm = OMatGRPOModule.__new__(OMatGRPOModule)
    L.LightningModule.__init__(lm)
    lm.fields = ("species",)
    lm.eps_clip = 0.2
    lm.beta_kl = 0.0
    lm.alpha_pos = 0.0
    lm.alpha_cell = 0.0
    lm.alpha_species = float(alpha_species)
    return lm


@pytest.mark.parametrize("alpha_species", [0.0, 0.1, 1.0])
def test_alpha_species_multiplies_species_term(alpha_species: float):
    """With new == old (ratio == 1 identically) and A == 1 uniformly, the
    species event term reduces to -1.0 analytically. The total policy loss
    must equal alpha_species * (-1.0) within numerical tolerance. This
    pins down α_species as a first-class multiplier on the species
    channel, matching α_pos and α_cell."""
    lm = _make_lm_with_alpha_species(alpha_species)

    BK, E = 4, 8
    # Identical logp_old and logp_new → ratio = 1 everywhere (before exp,
    # log_ratio = 0). We still need separate tensors so autograd sees
    # independent leaves.
    logp_vals = torch.randn(E)
    old_sp = {
        "logp":       logp_vals.clone(),
        "step_idx":   torch.zeros(E, dtype=torch.long),
        "struct_idx": torch.randint(0, BK, (E,), dtype=torch.long),
    }
    new_sp = {
        "logp":       logp_vals.clone(),
        "step_idx":   old_sp["step_idx"].clone(),
        "struct_idx": old_sp["struct_idx"].clone(),
    }
    logp_old = {"species": old_sp}
    logp_new = {"species": new_sp}

    advantages = torch.ones(BK)  # A == 1 → species_term = -mean(1*1) = -1

    loss, stats = lm._ppo_grpo_loss(
        logp_old, logp_new, advantages=advantages, analytical_kl=None,
    )

    expected = alpha_species * (-1.0)
    assert abs(float(loss) - expected) < 1e-6, (
        f"α_species={alpha_species} should give loss={expected}, got "
        f"{float(loss):.8f}. If alpha_species dropped out of the sum "
        f"the loss would be -1.0 regardless of α."
    )
    # Sanity: species event count recorded in stats.
    assert float(stats["stats/species_event_count"]) == float(E)


# =====================================================================
# T3 — Reward all-Z tolerance canary (_apply_mask_guard bifurcation)
# =====================================================================

def test_apply_mask_guard_all_Z_passes_through():
    """All-Z (no zeros) composition survives unchanged with has_mask=False.
    This is the decision that routes the structure to the UMA forward pass
    instead of the +inf sentinel — i.e. the case Phase-1-step-4 unlocked."""
    from omg.grpo.reward import _apply_mask_guard

    specie = np.array([1, 6, 8, 11, 26], dtype=np.int64)  # H, C, O, Na, Fe
    safe, has_mask = _apply_mask_guard(specie)

    assert has_mask is False, (
        "All-Z input must NOT be flagged as masked — it must reach UMA."
    )
    assert np.array_equal(safe, specie), (
        f"All-Z input must pass through unchanged; got {safe} vs {specie}. "
        f"If the guard rewrites non-zero species it will corrupt UMA "
        f"inputs."
    )


def test_apply_mask_guard_partial_mask_routes_to_sentinel():
    """Case (i) bifurcation canary. If any Z==0 the guard must substitute
    Z=1 placeholders (so downstream pymatgen Element.from_Z survives) and
    flag has_mask=True so calculate_batch_energy_reward routes to +inf."""
    from omg.grpo.reward import _apply_mask_guard

    specie = np.array([0, 6, 8], dtype=np.int64)  # mask + C + O
    safe, has_mask = _apply_mask_guard(specie)

    assert has_mask is True, (
        "Any Z==0 must flag has_mask=True for the +inf sentinel path."
    )
    assert np.array_equal(safe, np.array([1, 1, 1], dtype=np.int64)), (
        f"Partial-mask must be replaced with all-Z=1 placeholder; "
        f"got {safe}."
    )


def test_apply_mask_guard_empty_array():
    """Empty composition (zero-atom structure) must not crash — edge case
    from structure-level iteration in process_data."""
    from omg.grpo.reward import _apply_mask_guard

    specie = np.array([], dtype=np.int64)
    safe, has_mask = _apply_mask_guard(specie)

    assert has_mask is False
    assert safe.size == 0


# =====================================================================
# T4 — --alpha_species reaches OMatGRPOModule(alpha_species=...)
# =====================================================================

def test_alpha_species_flag_reaches_module_kwargs():
    from omg.grpo.train import module_kwargs, resolve_config
    cfg = resolve_config(["--alpha_species", "0.42"], env={})
    assert cfg["alpha_species"] == 0.42
    assert module_kwargs(cfg)["alpha_species"] == 0.42


# =====================================================================
# T5 — step_logprob(return_drift=True) skips empty drift_lists
# =====================================================================

def test_step_logprob_species_drift_empty_skip():
    """Regression canary for the CPU Trainer-harness crash:
    `torch.stack([])` raised RuntimeError when fields=('pos','cell','species')
    and return_drift=True, because species is DFM-discrete and never appends
    to drift_lists. Fix at stochastic_interpolants.py:793-800 filters empty
    lists before stacking.

    Two-pronged shield:
      (a) source-level — the `len(drift_lists[fname]) > 0` guard MUST stay in
          step_logprob's return_drift branch. A refactor that drops it
          reintroduces the crash.
      (b) behavioral — directly exercise the exact filter logic on a realistic
          drift_lists state (pos/cell populated, species empty) and assert the
          resulting `drifts` dict contains pos and cell but NOT species.

    Note: user brief said 'integrate_with_logprob' but that function does not
    accept return_drift; the return_drift=True branch lives in step_logprob."""
    # (a) Source-level shield.
    from omg.si.stochastic_interpolants import StochasticInterpolants

    src = inspect.getsource(StochasticInterpolants.step_logprob)
    assert re.search(r"len\(drift_lists\[.*?\]\)\s*>\s*0", src), (
        "step_logprob must guard empty drift_lists before torch.stack. "
        "Species is DFM-discrete and never appends a drift — without the "
        "`len(drift_lists[fname]) > 0` filter, `torch.stack([])` raises "
        "RuntimeError on the CPU Trainer harness."
    )

    # (b) Behavioral shield — reconstruct the exact filter inputs that surface
    # in a fields=('pos','cell','species') + return_drift=True call.
    T_steps, B = 3, 2
    drift_lists = {
        "pos": [torch.zeros(B, 3) for _ in range(T_steps)],
        "cell": [torch.zeros(B, 3, 3) for _ in range(T_steps)],
        "species": [],  # DFM-discrete — stays empty across all T inner steps.
    }
    fields = ("pos", "cell", "species")

    # Mirrors stochastic_interpolants.py:797-801 verbatim.
    drifts = {
        fname: torch.stack(drift_lists[fname], dim=0)
        for fname in fields
        if fname in drift_lists and len(drift_lists[fname]) > 0
    }

    assert "pos" in drifts, "pos drift must survive the filter."
    assert "cell" in drifts, "cell drift must survive the filter."
    assert "species" not in drifts, (
        "species drift must be filtered out — DFM-discrete has no drift. "
        "If species enters `drifts` with an empty list, `_compute_analytical_kl` "
        "downstream will crash on `torch.stack([])`."
    )
    assert drifts["pos"].shape == (T_steps, B, 3)
    assert drifts["cell"].shape == (T_steps, B, 3, 3)
