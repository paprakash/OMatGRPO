"""omg/grpo/guards.py is the single source of the cell-guard thresholds. It must classify every
cell exactly as the three earlier copies did (the reward's pre-relaxation guard, its
post-relaxation guard, and the evaluation script's guard), which are reimplemented here verbatim."""
import numpy as np
import pytest

from omg.grpo.guards import cell_ok, classify_cell


def _old_reward_bucket(cell_np):
    vol = abs(float(np.linalg.det(cell_np)))
    lens = np.linalg.norm(cell_np, axis=1)
    min_len, max_len = float(lens.min()), float(lens.max())
    a, b, c = cell_np[0], cell_np[1], cell_np[2]
    cross = np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)])
    min_perp = float((vol / np.maximum(np.linalg.norm(cross, axis=1), 1e-6)).min())
    if (vol < 1.0) or (min_len < 1.5):
        return "small"
    if min_perp < 1.5:
        return "oblate"
    if (vol > 5000.0) or (max_len > 50.0):
        return "huge"
    return "ok"


def _old_relaxed_cell_ok(cell_np):
    vol = abs(float(np.linalg.det(cell_np)))
    lens = np.linalg.norm(cell_np, axis=1)
    if not np.isfinite(cell_np).all():
        return False
    if vol < 1.0 or float(lens.min()) < 1.5:
        return False
    a, b, c = cell_np[0], cell_np[1], cell_np[2]
    cross = np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)])
    d_perps = vol / np.maximum(np.linalg.norm(cross, axis=1), 1e-6)
    if float(d_perps.min()) < 1.5:
        return False
    if vol > 5000.0 or float(lens.max()) > 50.0:
        return False
    return True


def _cells(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scale = rng.choice([0.05, 0.5, 2.0, 5.0, 20.0])
        out.append(rng.normal(size=(3, 3)) * scale + np.eye(3) * rng.uniform(0, 12))
    # edge cases: collapsed, flat, exactly at thresholds, non-finite
    out += [np.zeros((3, 3)), np.diag([1.5, 1.5, 1.5]), np.diag([1.49, 10, 10]),
            np.diag([10, 10, 50.0]), np.diag([10, 10, 50.01]), np.diag([20.0, 20.0, 12.5]),
            np.array([[5, 0, 0], [5, 0.1, 0], [0, 0, 5.0]]),
            np.array([[np.nan, 0, 0], [0, 5, 0], [0, 0, 5]]),
            np.array([[np.inf, 0, 0], [0, 5, 0], [0, 0, 5]])]
    return out


def test_classify_matches_old_reward_guard():
    for cell in _cells():
        if not np.isfinite(cell).all():
            continue   # the pre-relaxation guard never saw non-finite cells in isolation
        assert classify_cell(cell) == _old_reward_bucket(cell), cell


def test_cell_ok_matches_old_post_relax_guard():
    for cell in _cells(seed=1):
        with np.errstate(invalid="ignore"):
            assert cell_ok(cell) == _old_relaxed_cell_ok(cell), cell


@pytest.mark.parametrize("cell,bucket", [
    (np.diag([4.0, 4.0, 4.0]), "ok"),
    (np.diag([1.0, 4.0, 4.0]), "small"),
    (np.diag([60.0, 4.0, 4.0]), "huge"),
])
def test_examples(cell, bucket):
    assert classify_cell(cell) == bucket
