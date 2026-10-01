"""Smoke tests for omg.grpo.lemat_hull.

Validates the full HF download + parquet parse + composition filter +
PhaseDiagram-build path of get_energy_above_hull. The first call downloads
the LeMat-Bulk-MLIP-Hull parquet (~few hundred MB) and is cached via
lru_cache for subsequent calls.
"""
import pytest
from pymatgen.core import Composition

from omg.grpo.lemat_hull import get_energy_above_hull


@pytest.mark.network
@pytest.mark.parametrize("hull_type", ["uma", "dft"])
def test_lemat_hull_smoke_NaCl(hull_type):
    e_above = get_energy_above_hull(
        total_energy=-10.0,
        composition=Composition("NaCl"),
        hull_type=hull_type,
        threshold=0.001,
    )
    assert isinstance(e_above, float)


@pytest.mark.network
def test_lemat_hull_pure_element_falls_through():
    e_above = get_energy_above_hull(
        total_energy=-5.0,
        composition=Composition("Si"),
        hull_type="uma",
    )
    assert isinstance(e_above, float)
