"""
Bulk-crystal formation-energy references (used for formation-energy logging).

1. ``test_form_refs_match_lemat_genbench``: the local ``element_chem_pot.json`` matches
   known values of LeMat-GenBench's copy (PBE per-element lowest-bulk-energy references).
2. ``test_struct_00000_formation_energy``: the Nb4CO test structure gives
   0.4225 +/- 0.05 eV/atom, the reference value of LeMat-GenBench's
   ``UMACalculator.calculate_formation_energy``.

The second test loads UMA (about 30 s) and needs a CUDA device. Do not loosen the tolerance.
"""
import json
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = os.path.join(REPO, "omg", "grpo", "element_chem_pot.json")
CIF_PATH = os.path.join(REPO, "tests", "data", "struct_00000.cif")


def test_form_refs_match_lemat_genbench():
    assert os.path.exists(JSON_PATH), (
        f"element_chem_pot.json missing at {JSON_PATH} — pull from "
        "https://raw.githubusercontent.com/LeMaterial/lemat-genbench/main/"
        "src/lemat_genbench/preprocess/element_chem_pot.json"
    )
    with open(JSON_PATH) as f:
        refs = json.load(f)

    # Upstream LeMat-GenBench main as of 2026-05-05 ships 89 elements;
    # if the upstream coverage grows we just want to know about it, so
    # gate on a sane lower bound rather than an exact count.
    assert len(refs) >= 80, f"too few elements: {len(refs)}"

    # Spot-check three values pinned at pull time. Tolerance 1e-3 eV
    # absorbs serialization noise without hiding a real upstream change.
    expected = {"Nb": -10.1946, "C": -9.2325, "O": -4.9582}
    for sym, want in expected.items():
        assert sym in refs, f"missing element {sym}"
        got = refs[sym]["pbe"]
        assert abs(got - want) < 1e-3, f"{sym}: {got} vs expected {want}"


@pytest.mark.gpu
@pytest.mark.uma
@pytest.mark.network
def test_struct_00000_formation_energy():
    """Binding gate: 0.4225 ± 0.05 eV/atom for the Nb4CO test CIF.

    Requires CUDA + UMA model download. Guard with a CUDA-availability
    check so CPU-only environments don't fail on import-time skip.
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for UMA forward")

    assert os.path.exists(CIF_PATH), f"missing test CIF at {CIF_PATH}"

    from pymatgen.core import Structure
    from omg.grpo.reward import OMatGRPOReward

    struct = Structure.from_file(CIF_PATH)
    assert len(struct) == 12, f"expected 12 atoms, got {len(struct)}"

    reward = OMatGRPOReward(
        weights={"rmsd": 0.0, "energy": 1.0},
        reward_type="formation",
        device="cuda",
    )
    reward.calculate_batch_energy_reward([struct])
    fe = reward.last_formation_energy_per_atom[0]

    target, tol = 0.4225, 0.05
    assert target - tol <= fe <= target + tol, (
        f"struct_00000.cif E_form/atom = {fe:.4f}, "
        f"expected {target} ± {tol}. The bulk-crystal-refs fix may "
        "have regressed to iso-atom-refs (~ -6.83 eV/atom) or to a "
        "different reference convention."
    )
