"""
Per-channel log-probabilities of rollout and replay.

integrate_with_logprob (rollout) and step_logprob (replay) both return a dict:
  "pos":     FloatTensor [B, T]  per-step Gaussian log-probability, positions
  "cell":    FloatTensor [B, T]  per-step Gaussian log-probability, lattice
  "species": Dict[str, Tensor]   one entry per atom unmask event:
                logp:       FloatTensor [E]
                step_idx:   LongTensor  [E]   (values in [0, T))
                struct_idx: LongTensor  [E]   (values in [0, B))

Cases:
  1. test_species_event_shapes_rollout - E > 0 for an all-mask start; indices in range.
  2. test_species_rollout_replay_consistency - replay with the rollout model reproduces
     the rollout species log-probabilities (ratio 1 at the first inner epoch).
  3. test_ppo_loss_per_field_combiner - _ppo_grpo_loss accepts the dict, returns a scalar,
     and gradients flow through every channel.
"""
import torch

from tests.test_dfm_discrete_admission import (
    _ToyModelWithSpecies,
    _build_x0_masked,
    _build_si_with_species,
)


# =====================================================================
# Case 1 — event shape/index sanity
# =====================================================================

def test_species_event_shapes_rollout():
    """All-mask start, pos/cell-only fields: species must
    evolve every step and record >0 commit events across T steps. Each
    event carries a valid step_idx ∈ [0, T) and struct_idx ∈ [0, B)."""
    torch.manual_seed(1234)
    x0 = _build_x0_masked(seed=0)
    model = _ToyModelWithSpecies()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    si = _build_si_with_species(T=8)
    torch.manual_seed(42)
    gen, traj, logp = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=("pos", "cell"), stochastic=True, return_trajectory=True,
    )

    # Per-field dict shape
    assert isinstance(logp, dict)
    assert set(logp.keys()) >= {"pos", "cell", "species"}

    B = 2
    T_inner = logp["pos"].shape[1]
    assert logp["pos"].shape == (B, T_inner)
    assert logp["cell"].shape == (B, T_inner)

    sp = logp["species"]
    assert isinstance(sp, dict)
    assert set(sp.keys()) >= {"logp", "step_idx", "struct_idx"}
    E = sp["logp"].shape[0]
    assert E > 0, "an all-mask rollout must record at least one unmask event"
    assert sp["step_idx"].shape == (E,)
    assert sp["struct_idx"].shape == (E,)
    assert sp["step_idx"].dtype == torch.long
    assert sp["struct_idx"].dtype == torch.long
    assert int(sp["step_idx"].min()) >= 0 and int(sp["step_idx"].max()) < T_inner
    assert int(sp["struct_idx"].min()) >= 0 and int(sp["struct_idx"].max()) < B
    # logp values must be finite and non-positive (they are log-probs of a categorical).
    assert torch.isfinite(sp["logp"]).all()
    assert (sp["logp"] <= 0.0).all()

    # traj must store the final species state so step_logprob can diff at the last step.
    assert "species_final" in traj
    assert traj["species_final"].shape == x0.species.shape


# =====================================================================
# Case 2 — rollout-replay consistency (species epoch-0 invariant)
# =====================================================================

def test_species_rollout_replay_consistency():
    """When model ≡ model_ref (weights unchanged, epoch 0), species event
    logps must match between rollout (captured under species_source_model)
    and replay (computed under model). Same invariant as pos/cell
    epoch-0 consistency, just event-level."""
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
    gen, traj, logp_old = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=("pos", "cell"), stochastic=True, return_trajectory=True,
    )

    # Replay under the same model (weights unchanged)
    logp_new = si.step_logprob(model, traj, fields=("pos", "cell"))

    # Pos / cell (existing invariant, still holds per-field)
    assert logp_new["pos"].shape == logp_old["pos"].shape
    assert logp_new["cell"].shape == logp_old["cell"].shape
    assert torch.allclose(logp_new["pos"], logp_old["pos"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(logp_new["cell"], logp_old["cell"], atol=1e-5, rtol=1e-4)

    # Species event logps
    sp_old = logp_old["species"]
    sp_new = logp_new["species"]
    assert sp_new["logp"].shape == sp_old["logp"].shape, (
        f"species event count mismatch: rollout E={sp_old['logp'].shape[0]}, "
        f"replay E={sp_new['logp'].shape[0]}"
    )
    # step_idx and struct_idx must match identically (same commit events)
    assert torch.equal(sp_new["step_idx"], sp_old["step_idx"])
    assert torch.equal(sp_new["struct_idx"], sp_old["struct_idx"])
    max_diff = (sp_new["logp"] - sp_old["logp"]).abs().max().item()
    assert torch.allclose(sp_new["logp"], sp_old["logp"], atol=1e-5, rtol=1e-4), (
        f"Species epoch-0 invariant violated: max |Δlogp| = {max_diff:.3e}. "
        "Under model ≡ model_ref the rollout and replay logits should produce "
        "identical log_softmax values for each committed atom."
    )


# =====================================================================
# Case 3 — PPO loss per-field combiner shape + grad flow
# =====================================================================

def test_ppo_loss_per_field_combiner():
    """Build a bare OMatGRPOModule instance via __new__ (skip UMA load),
    feed a synthetic per-field dict, check the scalar loss runs and
    gradients propagate through each α weight."""
    from omg.grpo.grpo_lightning import OMatGRPOModule

    lightning = OMatGRPOModule.__new__(OMatGRPOModule)
    lightning.eps_clip = 0.2
    lightning.beta_kl = 0.0
    lightning.alpha_pos = 1.0
    lightning.alpha_cell = 1.0
    lightning.alpha_species = 1.0
    lightning.fields = ("pos", "cell", "species")
    lightning.k = 2

    BK, T, E = 4, 5, 7
    # _ppo_grpo_loss derives device from `advantages` input, so we don't need
    # to set lightning.device (which is a Lightning-managed read-only property).

    torch.manual_seed(0)
    logp_old = {
        "pos":  torch.randn(BK, T),
        "cell": torch.randn(BK, T),
        "species": {
            "logp":       torch.randn(E),
            "step_idx":   torch.randint(0, T, (E,), dtype=torch.long),
            "struct_idx": torch.randint(0, BK, (E,), dtype=torch.long),
        },
    }
    # logp_new carries grad; we need to see the loss depend on it.
    logp_new = {
        "pos":  (logp_old["pos"] + 0.01 * torch.randn(BK, T)).clone().requires_grad_(True),
        "cell": (logp_old["cell"] + 0.01 * torch.randn(BK, T)).clone().requires_grad_(True),
        "species": {
            "logp":       (logp_old["species"]["logp"] + 0.01 * torch.randn(E)).clone().requires_grad_(True),
            "step_idx":   logp_old["species"]["step_idx"],
            "struct_idx": logp_old["species"]["struct_idx"],
        },
    }
    advantages = torch.randn(BK)

    loss, stats = lightning._ppo_grpo_loss(
        logp_old, logp_new, advantages, analytical_kl=None,
    )
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss)

    loss.backward()
    assert logp_new["pos"].grad is not None and torch.isfinite(logp_new["pos"].grad).all()
    assert logp_new["cell"].grad is not None and torch.isfinite(logp_new["cell"].grad).all()
    assert logp_new["species"]["logp"].grad is not None
    assert torch.isfinite(logp_new["species"]["logp"].grad).all()
    # Per-field stat keys present
    assert "stats/ratio_mean_pos" in stats
    assert "stats/ratio_mean_cell" in stats
    assert "stats/ratio_mean_species" in stats
