"""
_rollout_groups with fields=("pos", "cell", "species").

  - When species is learned, the sampler's all-mask species reach x0_rep; otherwise the
    x1 composition is copied, so a group shares one fixed composition.
  - [A1] When species is learned, x0_rep starts all-mask and identical within a group.
  - [A2] When species is learned, generated compositions may differ within a group; fewer
    than ceil(K/4) distinct ones only give a warning.
  - [A3] The member-0 == x1 composition check applies only when species is not learned.

Cases:
  1. test_guard_keeps_all_mask_when_species_learned
  2. test_guard_preserves_x1_species_when_not_learned
  3. test_a1_rewrite_rejects_non_mask_when_species_learned
  4. test_a2_accepts_diverse_gen_species_when_species_learned
  5. test_a3_dropped_when_species_learned
"""
import pytest
import torch
import lightning as L

from omg.datamodule import OMGData
from omg.grpo.grpo_lightning import OMatGRPOModule


# =====================================================================
# Fixtures / stubs
# =====================================================================

def _make_x1(B: int = 2, n_per=(3, 3)) -> OMGData:
    """Minimal OMGData batch with non-mask species (real compositions)."""
    assert len(n_per) == B
    total = sum(n_per)
    n_atoms = torch.tensor(n_per, dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long), n_atoms)
    ptr = torch.zeros(B + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(n_atoms, dim=0)
    pos = torch.rand(total, 3)
    cell = torch.randn(B, 3, 3) * 0.3 + torch.eye(3).unsqueeze(0) * 5.0
    species = torch.randint(1, 50, (total,), dtype=torch.long)
    x = OMGData()
    x.pos = pos
    x.cell = cell
    x.species = species
    x.batch = batch
    x.n_atoms = n_atoms
    x.ptr = ptr
    return x


class _StubSampler:
    """Mimics sampler.sample_p_0(x1). species_mode controls whether the
    returned x0.species is all-M (MaskDistribution regime, the real
    species dist used by pretrained/train.yaml) or non-M."""

    def __init__(self, species_mode: str = "all_mask"):
        assert species_mode in ("all_mask", "non_mask")
        self.species_mode = species_mode

    def sample_p_0(self, x1: OMGData) -> OMGData:
        total = x1.pos.shape[0]
        x0 = OMGData()
        x0.pos = torch.rand_like(x1.pos)
        x0.cell = torch.randn_like(x1.cell)
        x0.batch = x1.batch.clone()
        x0.n_atoms = x1.n_atoms.clone()
        x0.ptr = x1.ptr.clone()
        if self.species_mode == "all_mask":
            x0.species = torch.zeros(total, dtype=torch.long)
        else:
            # Non-mask species, but NOT equal to x1.species byte-for-byte,
            # so the branch where x0 survives into x0_rep can be told apart
            # from the branch that copies x1.species.
            x0.species = torch.full((total,), 7, dtype=torch.long)
        return x0


class _StubSI:
    """Stubs si.integrate_with_logprob. Echoes x0_rep into gen; optionally
    mutates gen.species via `mutate_species(gen, x0_rep)` callback so each
    test can pin the post-integration species pattern it needs."""

    def __init__(self, mutate_species=None, T: int = 4):
        self.mutate_species = mutate_species
        self.T = T

    def integrate_with_logprob(self, x0_rep, model, model_ref, fields,
                               stochastic, return_trajectory):
        gen = OMGData()
        gen.pos = x0_rep.pos.clone()
        gen.cell = x0_rep.cell.clone()
        gen.batch = x0_rep.batch.clone()
        gen.n_atoms = x0_rep.n_atoms.clone()
        gen.ptr = x0_rep.ptr.clone()
        gen.species = x0_rep.species.clone()
        if self.mutate_species is not None:
            self.mutate_species(gen, x0_rep)
        BK = int(gen.n_atoms.numel())
        logp = {
            "pos":  torch.randn(BK, self.T),
            "cell": torch.randn(BK, self.T),
            "species": {
                "logp":       torch.zeros(0),
                "step_idx":   torch.zeros(0, dtype=torch.long),
                "struct_idx": torch.zeros(0, dtype=torch.long),
            },
        }
        traj = {"species_final": gen.species.clone()}
        return gen, traj, logp


def _make_lightning(
    fields, k: int = 4, sampler_mode: str = "all_mask", mutate_species=None
) -> OMatGRPOModule:
    """Bare OMatGRPOModule with stubbed sampler + SI, ready for
    _rollout_groups invocation on CPU."""
    lm = OMatGRPOModule.__new__(OMatGRPOModule)
    L.LightningModule.__init__(lm)
    lm.fields = tuple(fields)
    lm.k = k
    lm.dng_mode = False
    lm.sampler = _StubSampler(species_mode=sampler_mode)
    lm.si = _StubSI(mutate_species=mutate_species)
    lm.model = torch.nn.Module()
    lm.model_ref = torch.nn.Module()
    return lm


# =====================================================================
# Case 1 — x0.species stays all-mask when species ∈ fields
# =====================================================================

def test_guard_keeps_all_mask_when_species_learned():
    """Post-Task-C guard at L314: with species ∈ fields, the sampler's
    all-M species must survive into x0_rep. Pre-Task-C the unconditional
    overwrite flips x0.species to x1.species (non-M)."""
    torch.manual_seed(0)
    x1 = _make_x1(B=2, n_per=(3, 3))
    k = 4

    # Make gen.species = 0 (echo all-M). With k=4, ceil(k/4)=1 unique per
    # group, so the post-Task-C diversity check (≥ ceil(k/4)) accepts this.
    def _echo_all_mask(gen, x0_rep):
        gen.species = torch.zeros_like(gen.species)

    lm = _make_lightning(
        fields=("pos", "cell", "species"), k=k,
        sampler_mode="all_mask", mutate_species=_echo_all_mask,
    )
    x0_rep, gen, traj, logp = lm._rollout_groups(x1)

    assert (x0_rep.species == 0).all(), (
        "x0_rep.species must stay all-mask when "
        "species ∈ fields. Got non-mask tokens: "
        f"unique={x0_rep.species.unique().tolist()}"
    )


# =====================================================================
# Case 2 — canary: overwrite still fires when species ∉ fields
# =====================================================================

def test_guard_preserves_x1_species_when_not_learned():
    """When species is not learned, x0.species is overwritten with x1.species so
    _repeat_groups replicates one fixed composition across the group."""
    torch.manual_seed(0)
    x1 = _make_x1(B=2, n_per=(3, 3))
    k = 4

    def _echo(gen, x0_rep):
        pass  # default echo of x0_rep.species

    lm = _make_lightning(
        fields=("pos", "cell"), k=k,
        sampler_mode="all_mask", mutate_species=_echo,
    )
    x0_rep, gen, traj, logp = lm._rollout_groups(x1)

    # x0_rep.species must equal x1.species repeated K times per group.
    for g in range(2):
        a1 = int(x1.ptr[g].item())
        b1 = int(x1.ptr[g + 1].item())
        n_g = b1 - a1
        x1_slice = x1.species[a1:b1]
        a_start = int(x0_rep.ptr[g * k].item())
        group = x0_rep.species[a_start:a_start + k * n_g].view(k, n_g)
        assert torch.equal(group[0], x1_slice), (
            f"canary: group g={g} member 0 must match x1.species[{a1}:{b1}] "
            "when species ∉ fields"
        )
        assert (group == group[0:1]).all(), (
            f"canary: group g={g} must stay K-identical when species ∉ fields"
        )


# =====================================================================
# Case 3 — [A1] rewrite rejects non-M x0_rep when species ∈ fields
# =====================================================================

def test_a1_rewrite_rejects_non_mask_when_species_learned():
    """Post-Task-C [A1] rewrite: when species ∈ fields, x0_rep.species
    must be all-M (sampler's MaskDistribution output). Inject non-M via
    sampler stub, so [A1] raises AssertionError."""
    torch.manual_seed(0)
    x1 = _make_x1(B=2, n_per=(3, 3))
    k = 4

    def _echo(gen, x0_rep):
        pass

    lm = _make_lightning(
        fields=("pos", "cell", "species"), k=k,
        sampler_mode="non_mask",  # sampler returns Z=7 everywhere
        mutate_species=_echo,
    )

    with pytest.raises(AssertionError, match=r"all[- ]?M|mask|\[A1\]"):
        lm._rollout_groups(x1)


# =====================================================================
# Case 4 — [A2] accepts diverse gen.species when species ∈ fields
# =====================================================================

def test_a2_accepts_diverse_gen_species_when_species_learned():
    """Post-Task-C [A2] rewrite: when species ∈ fields, gen.species
    need not be K-identical within a group; a diversity check of
    n_unique ≥ ceil(K/4) is sufficient. Here K=8 ⇒ threshold=2 and we
    inject 8 distinct per-member compositions."""
    torch.manual_seed(0)
    x1 = _make_x1(B=2, n_per=(3, 3))
    k = 8

    def _mutate_diverse(gen, x0_rep):
        # Set each member m of each group to composition (m+1, m+1, ...)
        ptr = x0_rep.ptr
        BK = int(x0_rep.n_atoms.numel())
        for m in range(BK):
            a, b = int(ptr[m].item()), int(ptr[m + 1].item())
            # member index within group = m % k
            gen.species[a:b] = (m % k) + 1

    lm = _make_lightning(
        fields=("pos", "cell", "species"), k=k,
        sampler_mode="all_mask", mutate_species=_mutate_diverse,
    )
    x0_rep, gen, traj, logp = lm._rollout_groups(x1)

    # Verify diversity post-rollout: each group must have ≥ ceil(k/4) unique.
    threshold = (k + 3) // 4
    for g in range(2):
        a_start = int(gen.ptr[g * k].item())
        n_g = int(gen.n_atoms[g * k].item())
        group = gen.species[a_start:a_start + k * n_g].view(k, n_g)
        comps = {tuple(sorted(group[i].tolist())) for i in range(k)}
        assert len(comps) >= threshold, (
            f"[A2] diversity post-Task-C: group g={g} has {len(comps)} unique "
            f"compositions, need ≥ {threshold}"
        )


# =====================================================================
# Case 5 — [A3] dropped when species ∈ fields (member-0 may differ from x1)
# =====================================================================

def test_a3_dropped_when_species_learned():
    """Post-Task-C [A3] drop: when species ∈ fields, gen.species's
    member-0 is NOT required to byte-equal x1.species (we sampled a
    new composition — that's the whole point).

    Inject gen.species = 42 everywhere (K-identical per group, but
    member-0 ≠ x1.species). Post-Task-C: [A3] check is wrapped behind
    `if "species" not in self.fields:` and skipped → success."""
    torch.manual_seed(0)
    x1 = _make_x1(B=2, n_per=(3, 3))
    k = 4

    def _mutate_mismatch(gen, x0_rep):
        gen.species = torch.full_like(gen.species, 42)

    lm = _make_lightning(
        fields=("pos", "cell", "species"), k=k,
        sampler_mode="all_mask", mutate_species=_mutate_mismatch,
    )
    x0_rep, gen, traj, logp = lm._rollout_groups(x1)

    assert (gen.species == 42).all(), "stub mutate must set gen.species to 42"
    # And crucially: gen.species[0] for each group does NOT match x1.species
    for g in range(2):
        a1 = int(x1.ptr[g].item())
        b1 = int(x1.ptr[g + 1].item())
        x1_slice = x1.species[a1:b1]
        a_start = int(gen.ptr[g * k].item())
        member0 = gen.species[a_start:a_start + (b1 - a1)]
        assert not torch.equal(member0, x1_slice), (
            f"test pre-condition: group g={g} member 0 must differ from x1.species "
            "to prove [A3] was dropped (if they matched, we couldn't tell)"
        )
