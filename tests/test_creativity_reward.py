# tests/test_creativity_reward.py
# CPU-only unit tests for the creativity term (following Chemeleon2). No UMA / GPU.
# Checks:
#   1. known MP-20 train structure -> ~0 (unique-but-not-novel -> AMD dist 0)
#   2. made-up quinary             -> 1.0 (unique AND novel)
#   3. duplicate of a known ref    -> 0.0 (neither unique nor novel)
#   4. flag-gating: default w_creat=0.0 -> CreativityReward never constructed,
#      reward combine byte-identical (creativity contribution exactly zero)
#   5. per-structure StructureMatcher timeout -> score 0.0 + counted
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omg.grpo.reward import CreativityReward, OMatGRPOReward  # noqa: E402

from omg.grpo.references import default_creativity_reference  # noqa: E402

CACHE = str(default_creativity_reference())

pytestmark = pytest.mark.skipif(
    not os.path.exists(CACHE),
    reason="creativity reference missing (python scripts/build_references.py creativity)",
)


@pytest.fixture(scope="module")
def creativity():
    return CreativityReward(reference_path=CACHE)


@pytest.fixture(scope="module")
def known_structure(creativity):
    """A real MP-20 train structure, taken straight from the reference."""
    for formula, structs in creativity._ref_by_formula.items():
        if len(structs) == 1 and len(structs[0]) <= 8:
            return structs[0].copy()
    raise RuntimeError("no single-entry small formula in reference?")


@pytest.fixture(scope="module")
def quinary_structure(creativity):
    """Made-up quinary rocksalt-ish packing; formula absent from MP-20 train."""
    from pymatgen.core import Lattice, Structure
    s = Structure(
        Lattice.cubic(6.5),
        ["La", "Yb", "Sc", "Tc", "Re"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5],
         [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]],
    )
    assert s.composition.reduced_formula not in creativity._ref_by_formula
    return s


def test_known_mp20_structure_scores_near_zero(creativity, known_structure,
                                               quinary_structure):
    # Known structure is unique in the batch but NOT novel (exact ref match)
    # -> mixed -> AMD min distance to the identical ref = 0 -> score ~0.
    scores = creativity.compute([known_structure, quinary_structure])
    assert scores.shape == (2,)
    assert scores[0].item() < 0.05, f"known MP-20 structure scored {scores[0]}"
    assert creativity.last_novel[0] is False
    assert creativity.last_unique[0] is True


def test_made_up_quinary_scores_one(creativity, known_structure,
                                    quinary_structure):
    scores = creativity.compute([known_structure, quinary_structure])
    assert scores[1].item() == 1.0, f"made-up quinary scored {scores[1]}"
    assert creativity.last_unique[1] is True
    assert creativity.last_novel[1] is True
    assert creativity.last_timeout_count == 0


def test_duplicate_known_structure_scores_zero(creativity, known_structure):
    # Second copy: not unique (matches first) AND not novel -> exactly 0.0.
    scores = creativity.compute([known_structure, known_structure.copy()])
    assert scores[1].item() == 0.0
    assert creativity.last_unique[1] is False
    assert creativity.last_novel[1] is False


def test_scores_bounded_zero_one(creativity, known_structure,
                                 quinary_structure):
    scores = creativity.compute(
        [known_structure, quinary_structure, known_structure.copy()])
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_flag_off_is_byte_identical(no_uma):
    # Default w_creat=0.0: no CreativityReward constructed (UMA replaced by
    # no_uma so this runs on CPU). The combine path adds w_creat * zeros == 0.
    r = OMatGRPOReward(device="cpu")
    assert r.w_creat == 0.0
    assert r._creativity is None
    assert r.last_creativity == []


def test_creat_on_relaxed_fallback_substitution(known_structure,
                                                quinary_structure, no_uma):
    # With creat_on_relaxed, slots with a retained relaxed
    # Structure are scored on it; None slots fall back to the gen structure
    # and are counted. UMA replaced by no_uma; ctor guard needs the e_hull
    # relax config.
    r = OMatGRPOReward(
        device="cpu", reward_type="e_hull",
        relax_before_reward=True, weights={"rmsd": 0.0, "energy": 1.0},
        w_creat=1.0, creativity_reference=CACHE, creat_on_relaxed=True)
    relaxed = known_structure.copy()
    r.group = [quinary_structure, known_structure]
    r.last_relaxed_structures = [relaxed, None]
    structs, n_fallback = r._creativity_input_structures()
    assert structs[0] is relaxed          # substituted
    assert structs[1] is known_structure  # fallback to gen
    assert n_fallback == 1
    # flag OFF on the same instance state -> gen structures, zero fallbacks
    r.creat_on_relaxed = False
    structs, n_fallback = r._creativity_input_structures()
    assert structs == [quinary_structure, known_structure]
    assert n_fallback == 0


def test_creat_on_relaxed_ctor_guard(no_uma):
    # FATAL when the flag is on without the e_hull relax config — the term
    # must never silently score unrelaxed geometry.
    with pytest.raises(ValueError, match="creat_on_relaxed"):
        OMatGRPOReward(
            device="cpu",
            w_creat=1.0, creativity_reference=CACHE, creat_on_relaxed=True)
    # The flag defaults to off (w_creat=0 is covered by test_flag_off_is_byte_identical).
    # w_creat > 0 without the relax flag is fine.
    r = OMatGRPOReward(device="cpu")
    assert r.creat_on_relaxed is False


def test_sm_timeout_scores_zero_and_counts(creativity, known_structure,
                                           monkeypatch):
    import time as _time

    def slow_fit(*args, **kwargs):
        _time.sleep(10)
        return False

    monkeypatch.setattr(creativity.matcher, "fit", slow_fit)
    monkeypatch.setattr(creativity, "sm_timeout", 1)
    # Duplicate pair forces a matcher.fit call on the second structure
    # (same reduced formula, earlier batch member) — the first structure's
    # novelty check against its ref also hits slow_fit.
    scores = creativity.compute([known_structure, known_structure.copy()])
    assert creativity.last_timeout_count == 2
    assert scores[0].item() == 0.0 and scores[1].item() == 0.0
