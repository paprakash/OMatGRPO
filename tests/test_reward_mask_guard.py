"""
Masked-species guard in omg/grpo/reward.py.

A structure that still carries the mask token Z=0 would crash Element.from_Z(0) in
process_data. The guard replaces the tokens with Z=1 placeholders and flags the structure,
which is then not scored, like small/oblate/huge cells (see _route_and_clamp_ehull for
what that means for the reward).

Cases:
  1-4. test_apply_mask_guard_* - unit tests of the numpy helper.
  5.   test_process_data_sets_last_has_mask - integration, without loading UMA
       (OMatGRPOReward.__new__).
"""
import numpy as np
import torch
from torch_geometric.data import Data

from omg.grpo.reward import _apply_mask_guard, OMatGRPOReward


# =====================================================================
# Cases 1-4 — helper unit tests
# =====================================================================

def test_apply_mask_guard_no_mask():
    arr = np.array([1, 2, 3], dtype=np.int64)
    safe, has_mask = _apply_mask_guard(arr)
    np.testing.assert_array_equal(safe, arr)
    assert has_mask is False


def test_apply_mask_guard_some_masked():
    arr = np.array([0, 1, 2], dtype=np.int64)
    safe, has_mask = _apply_mask_guard(arr)
    np.testing.assert_array_equal(safe, np.array([1, 1, 1], dtype=np.int64))
    assert has_mask is True


def test_apply_mask_guard_all_masked():
    arr = np.array([0, 0, 0], dtype=np.int64)
    safe, has_mask = _apply_mask_guard(arr)
    np.testing.assert_array_equal(safe, np.array([1, 1, 1], dtype=np.int64))
    assert has_mask is True


def test_apply_mask_guard_empty():
    arr = np.array([], dtype=np.int64)
    safe, has_mask = _apply_mask_guard(arr)
    np.testing.assert_array_equal(safe, arr)
    assert has_mask is False


# =====================================================================
# Case 5 — process_data integration, UMA load bypassed
# =====================================================================

def test_process_data_sets_last_has_mask():
    """Build a reward without invoking __init__ (skips UMA load + HF fetch +
    CUDA init). Feed a synthetic gen with one masked and one clean structure.
    process_data must populate self.last_has_mask = [True, False] and must not
    crash on the masked structure."""
    reward = OMatGRPOReward.__new__(OMatGRPOReward)

    # Two structures, 2 atoms each. Structure 0 has a Z=0 mask token.
    pos = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.1, 0.2],
        [0.3, 0.4, 0.5],
    ], dtype=torch.float32)
    species = torch.tensor([0, 6, 14, 8], dtype=torch.long)  # struct 0 masked, struct 1 clean
    cell = torch.tensor([
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    ], dtype=torch.float32)
    n_atoms = torch.tensor([2, 2], dtype=torch.long)
    gen = Data(pos=pos, species=species, cell=cell, n_atoms=n_atoms)

    atoms, structures = reward.process_data(gen)

    assert len(structures) == 2
    assert len(atoms) == 2
    assert hasattr(reward, "last_has_mask"), "process_data must set self.last_has_mask"
    assert reward.last_has_mask == [True, False]
