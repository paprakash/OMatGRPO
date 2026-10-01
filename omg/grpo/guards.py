"""Guard thresholds shared by the reward and the evaluation scripts (one source of truth).

Cell guard: a stochastic lattice channel can produce cells that UMA cannot score.
  small  : volume below DEGEN_VOL_THRESHOLD or a lattice vector shorter than DEGEN_LEN_THRESHOLD
           (the periodic replica count blows up).
  oblate : a perpendicular lattice-plane distance d_perp_i = V / |a_j x a_k| below
           OBLATE_PERP_THRESHOLD (UMA's replica count scales with it, so such a cell can exhaust
           GPU memory even when it passes the other checks).
  huge   : volume above HUGE_VOL_THRESHOLD or a vector longer than HUGE_LEN_THRESHOLD
           (far outside MP-20, physically meaningless, can exhaust GPU memory).
Priority: small > oblate > huge > ok.

Hull trust thresholds (defaults of the training reward and flags of the evaluation):
  EHULL_MAG_FLOOR  : E_hull below this (eV/atom) counts as deep below the hull.
  EHULL_MIN_REFSET : fewer hull reference entries than this counts as a sparse hull.
"""
from typing import Tuple

import numpy as np

DEGEN_VOL_THRESHOLD = 1.0      # A^3; physical cells are >> 10 A^3/atom
DEGEN_LEN_THRESHOLD = 1.5      # A;   shortest realistic vector ~2-3 A
HUGE_VOL_THRESHOLD = 5000.0    # A^3; MP-20 tail ~500, 10x margin
HUGE_LEN_THRESHOLD = 50.0      # A;   MP-20 longest vector ~12, 4x margin
OBLATE_PERP_THRESHOLD = 1.5    # A;   perpendicular lattice-plane distance

EHULL_MAG_FLOOR = -0.1
EHULL_MIN_REFSET = 12


def cell_geometry(cell_np) -> Tuple[float, float, float, float]:
    """(volume, shortest vector, longest vector, smallest perpendicular plane distance)."""
    vol = abs(float(np.linalg.det(cell_np)))
    lens = np.linalg.norm(cell_np, axis=1)
    a, b, c = cell_np[0], cell_np[1], cell_np[2]
    cross = np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)])
    # Guard zero cross products (fully collapsed cells are already caught by the volume check)
    d_perps = vol / np.maximum(np.linalg.norm(cross, axis=1), 1e-6)
    return vol, float(lens.min()), float(lens.max()), float(d_perps.min())


def classify_cell(cell_np) -> str:
    """'small', 'oblate', 'huge' or 'ok' (no finiteness check; see cell_ok)."""
    vol, min_len, max_len, min_perp = cell_geometry(cell_np)
    if vol < DEGEN_VOL_THRESHOLD or min_len < DEGEN_LEN_THRESHOLD:
        return "small"
    if min_perp < OBLATE_PERP_THRESHOLD:
        return "oblate"
    if vol > HUGE_VOL_THRESHOLD or max_len > HUGE_LEN_THRESHOLD:
        return "huge"
    return "ok"


def cell_ok(cell_np) -> bool:
    """True if the cell is finite and passes the cell guard (used after a relaxation)."""
    if not np.isfinite(cell_np).all():
        return False
    return classify_cell(cell_np) == "ok"
