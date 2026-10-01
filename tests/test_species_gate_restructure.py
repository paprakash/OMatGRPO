"""
Species evolution in omg/si/stochastic_interpolants.py::integrate_with_logprob.

Species evolve at every step whenever a species interpolant is present and the species
start fully masked (Z=0), whether or not "species" is a learned field. Otherwise the
model would see positions and lattice evolving next to an all-mask composition, a joint
state it never saw in pretraining. Three checks keep the semantics explicit:
  - species in fields requires an all-mask start (else ValueError)
  - mixed mask states (some Z=0, some non-zero) are rejected (ValueError)
  - evolving species that are not learned requires model_ref (AssertionError)

Cases:
  1. test_species_evolves_with_all_masked_no_fields
  2. test_species_pinned_when_not_masked
  3. test_species_in_fields_requires_all_masked
  4. test_mixed_mask_states_rejected
  5. test_missing_model_ref_rejected
"""
import pytest
import torch

from tests.test_dfm_discrete_admission import (
    _ToyModelWithSpecies,
    _build_x0_masked,
    _build_si_with_species,
)


def _build_x0_resolved(seed: int = 0):
    """x0 with resolved species (no masks): the fixed-composition regime."""
    x0 = _build_x0_masked(seed=seed)
    total_atoms = x0.species.shape[0]
    torch.manual_seed(seed + 99)
    x0.species = torch.randint(1, 50, (total_atoms,), dtype=torch.long)
    return x0


def _build_x0_mixed(seed: int = 0):
    """x0 where some atoms are masked (Z=0) and some are not — mixed state."""
    x0 = _build_x0_masked(seed=seed)
    total_atoms = x0.species.shape[0]
    sp = torch.zeros(total_atoms, dtype=torch.long)
    sp[0] = 6
    x0.species = sp
    return x0


def _model_nograd():
    model = _ToyModelWithSpecies()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =====================================================================
# Case 1 — happy path: all-M start, no species in fields, species evolves
# =====================================================================

def test_species_evolves_with_all_masked_no_fields():
    """All-mask start, fields=('pos','cell'): species must evolve every step under model_ref, so the final species
    has at least one unmasked atom and species_seq is not all-zero at the
    final stored step."""
    torch.manual_seed(1234)
    x0 = _build_x0_masked(seed=0)
    assert (x0.species == 0).all(), "precondition: all species masked"

    model = _model_nograd()
    si = _build_si_with_species(T=8)

    torch.manual_seed(42)
    gen, traj, logp = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=("pos", "cell"), stochastic=True, return_trajectory=True,
    )

    assert (traj["species_seq"][0] == 0).all(), \
        "species_seq[0] should be the initial (all-M) state"
    assert (traj["species_seq"][-1] != 0).any(), \
        "after step 2 the final stored species_seq should show unmasking"
    assert (gen.species != 0).any(), \
        "final evolved species must have at least one unmasked atom"


# =====================================================================
# Case 2 — pre-resolved species stays pinned
# =====================================================================

def test_species_pinned_when_not_masked():
    """Fixed-composition regime: x0 species resolved, fields=('pos','cell').
    Species must stay pinned (no DFM evolution)."""
    torch.manual_seed(1234)
    x0 = _build_x0_resolved(seed=0)
    initial_species = x0.species.clone()
    assert (initial_species != 0).all(), "precondition: no species masked"

    model = _model_nograd()
    si = _build_si_with_species(T=8)

    torch.manual_seed(42)
    gen, traj, logp = si.integrate_with_logprob(
        x0, model, model_ref=model,
        fields=("pos", "cell"), stochastic=True, return_trajectory=True,
    )

    assert torch.equal(gen.species, initial_species), \
        "pre-resolved species must remain unchanged when not all-masked"
    for j in range(traj["species_seq"].shape[0]):
        assert torch.equal(traj["species_seq"][j], initial_species), \
            f"species_seq[{j}] must equal initial species (pinned)"


# =====================================================================
# Case 3 — guardrail: species in fields requires all-M start
# =====================================================================

def test_species_in_fields_requires_all_masked():
    """Caller asks to LEARN species via 'species' in fields, but x0 already has
    resolved species → misconfiguration, must raise ValueError."""
    torch.manual_seed(1234)
    x0 = _build_x0_resolved(seed=0)
    model = _model_nograd()
    si = _build_si_with_species(T=8)

    with pytest.raises(ValueError, match="all-M start|pre-resolved"):
        si.integrate_with_logprob(
            x0, model, model_ref=model,
            fields=("pos", "cell", "species"), stochastic=True,
            return_trajectory=True,
        )


# =====================================================================
# Case 4 — guardrail: mixed mask states rejected
# =====================================================================

def test_mixed_mask_states_rejected():
    """Partial mask (some Z=0, some not) is neither an all-mask start nor a fixed composition —
    must raise ValueError."""
    torch.manual_seed(1234)
    x0 = _build_x0_mixed(seed=0)
    model = _model_nograd()
    si = _build_si_with_species(T=8)

    with pytest.raises(ValueError, match="mixed mask states"):
        si.integrate_with_logprob(
            x0, model, model_ref=model,
            fields=("pos", "cell"), stochastic=True,
            return_trajectory=True,
        )


# =====================================================================
# Case 5 — guardrail: missing model_ref with an all-mask start and species not learned
# =====================================================================

def test_missing_model_ref_rejected():
    """All-mask start with species not in fields requires
    model_ref to supply the species source. Missing model_ref must assert."""
    torch.manual_seed(1234)
    x0 = _build_x0_masked(seed=0)
    model = _model_nograd()
    si = _build_si_with_species(T=8)

    with pytest.raises(AssertionError, match="needs model_ref"):
        si.integrate_with_logprob(
            x0, model, model_ref=None,
            fields=("pos", "cell"), stochastic=True,
            return_trajectory=True,
        )
