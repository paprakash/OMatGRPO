"""Hull lookup in the reward (OMatGRPOReward._lookup_e_hull): failures give NaN and a
reference count of 0 as before, are counted, and a batch in which every attempted lookup fails
raises instead of silently training without a stability signal."""
import math

import pytest
from ase import Atoms

import omg.grpo.lemat_hull as lemat_hull
from omg.grpo.reward import OMatGRPOReward


def _bare():
    return OMatGRPOReward.__new__(OMatGRPOReward)


def _atoms(symbols):
    return Atoms(symbols, cell=[4, 4, 4], pbc=True)


def _fake_lookup(fail_formulas):
    def lookup(e_total, comp, hull_type, threshold, return_ref_count):
        assert hull_type == "uma" and threshold == 0.001 and return_ref_count
        if comp.reduced_formula in fail_formulas:
            raise ValueError("no hull for " + comp.reduced_formula)
        return 0.05, 20
    return lookup


def test_partial_failure_is_nan_and_counted(monkeypatch):
    monkeypatch.setattr(lemat_hull, "get_energy_above_hull", _fake_lookup({"CsF"}))
    r = _bare()
    atoms = [_atoms("NaCl"), _atoms("CsF"), _atoms("KBr"), _atoms("MgO")]
    eh, ref = r._lookup_e_hull(atoms, ok_idx=[0, 1, 3], e_list=[-5.0, -4.0, -3.0, float("inf")])
    # 0: looked up; 1: lookup failed; 2: outside ok_idx; 3: non-finite energy
    assert eh[0] == 0.05 and ref[0] == 20
    assert math.isnan(eh[1]) and ref[1] == 0
    assert math.isnan(eh[2]) and ref[2] == 0
    assert math.isnan(eh[3]) and ref[3] == 0
    assert r.last_hull_lookup_attempted == 2
    assert r.last_hull_lookup_failures == 1


def test_whole_batch_failure_raises(monkeypatch):
    monkeypatch.setattr(lemat_hull, "get_energy_above_hull", _fake_lookup({"NaCl", "KBr"}))
    r = _bare()
    with pytest.raises(RuntimeError, match="failed for all 2 structures"):
        r._lookup_e_hull([_atoms("NaCl"), _atoms("KBr")], ok_idx=[0, 1], e_list=[-5.0, -3.0])


def test_no_attempt_does_not_raise(monkeypatch):
    """A batch where every structure failed a guard never reaches the lookup; that is not a
    hull-reference problem and must not raise."""
    monkeypatch.setattr(lemat_hull, "get_energy_above_hull", _fake_lookup(set()))
    r = _bare()
    eh, ref = r._lookup_e_hull([_atoms("NaCl")], ok_idx=[], e_list=[float("inf")])
    assert math.isnan(eh[0]) and ref == [0]
    assert r.last_hull_lookup_attempted == 0
