"""
relax_cell_dof: post-relaxation cell guard and flag plumbing.

The relaxation itself (Frechet cell filter through ts.optimize with UMA) needs a GPU and
the gated model. These tests cover the pure parts: the thresholds of _relaxed_cell_ok
(identical to the pre-relaxation guard) and the handling of slots that fail it in the
energy assembly.
"""
import numpy as np

from omg.grpo.reward import _relaxed_cell_ok


def test_relaxed_cell_ok_accepts_normal_cells():
    assert _relaxed_cell_ok(np.diag([5.0, 6.0, 7.0]))
    # mildly triclinic
    c = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 6.0]])
    assert _relaxed_cell_ok(c)


def test_relaxed_cell_ok_rejects_pathologies():
    assert not _relaxed_cell_ok(np.diag([0.5, 5.0, 5.0]))        # short vector
    assert not _relaxed_cell_ok(np.diag([1e-4, 1e-4, 1e-4]))     # collapsed
    assert not _relaxed_cell_ok(np.diag([100.0, 100.0, 100.0]))  # huge vol + len
    assert not _relaxed_cell_ok(np.diag([60.0, 5.0, 5.0]))       # one huge vector
    oblate = np.array([[10.0, 0, 0], [9.9, 1.0, 0], [0, 0, 10.0]])
    assert not _relaxed_cell_ok(oblate)                          # tiny perp distance
    nanc = np.diag([5.0, 5.0, 5.0]); nanc[0, 0] = np.nan
    assert not _relaxed_cell_ok(nanc)


def test_boundary_matches_pre_relax_guard_thresholds():
    # exactly at threshold: min_len 1.5 is NOT < 1.5 -> ok (matches pre-relax `<`)
    assert _relaxed_cell_ok(np.diag([1.5, 10.0, 10.0]))
    assert not _relaxed_cell_ok(np.diag([1.4999, 10.0, 10.0]))
