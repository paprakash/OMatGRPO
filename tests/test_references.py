"""omg/grpo/references.py: composition matrix from an LMDB, and the creativity reference in its
released JSON format, which must load into Structures bit-identical to the pickle used in the paper."""
import pickle

import lmdb
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from omg.grpo import references as refs
from omg.grpo.reward import CreativityReward


def _write_lmdb(path, records):
    env = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, rec in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(rec))
    env.close()


def test_composition_matrix_from_lmdb(tmp_path):
    path = tmp_path / "train.lmdb"
    _write_lmdb(path, [{"atomic_numbers": np.array([11, 17])},
                       {"atomic_numbers": np.array([8, 8, 12])}])
    M = refs.composition_matrix_from_lmdb(path, dim=119)
    assert M.shape == (2, 119)
    assert M[0, 11] == 0.5 and M[0, 17] == 0.5
    assert M[1, 8] == pytest.approx(2 / 3) and M[1, 12] == pytest.approx(1 / 3)
    assert np.allclose(M.sum(axis=1), 1.0)


def _structures():
    s1 = Structure(Lattice.from_parameters(4.1, 4.2, 4.3, 89.5, 90.3, 91.0), ["Na", "Cl"],
                   [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5000000000001]])
    s2 = Structure(Lattice.cubic(3.9), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return {s1.composition.reduced_formula: [s1], s2.composition.reduced_formula: [s2]}


def test_json_round_trip_is_bit_exact(tmp_path):
    ref = _structures()
    path = tmp_path / refs.CREATIVITY_REFERENCE_NAME
    refs.save_creativity_reference(ref, path)
    back = refs.load_creativity_reference(path)
    assert list(back) == list(ref)
    for k in ref:
        for a, b in zip(ref[k], back[k]):
            assert np.array_equal(a.lattice.matrix, b.lattice.matrix)
            assert np.array_equal(a.frac_coords, b.frac_coords)
            assert [str(x) for x in a.species] == [str(x) for x in b.species]


def test_json_file_is_reproducible(tmp_path):
    ref = _structures()
    refs.save_creativity_reference(ref, tmp_path / "a.json.gz")
    refs.save_creativity_reference(ref, tmp_path / "b.json.gz")
    assert (tmp_path / "a.json.gz").read_bytes() == (tmp_path / "b.json.gz").read_bytes()


def test_pickle_still_loads(tmp_path):
    ref = _structures()
    path = tmp_path / "ref.pkl"
    with open(path, "wb") as f:
        pickle.dump(ref, f)
    assert refs.load_creativity_reference(path).keys() == ref.keys()


def test_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_references.py creativity"):
        CreativityReward(reference_path=tmp_path / "missing.json.gz")
