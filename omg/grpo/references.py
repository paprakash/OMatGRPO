"""The two MP-20 references of the reward: the composition matrix of the MMD bonus and the
structures of the creativity term. Built by scripts/build_references.py into
<data_dir>/references (omg/grpo/paths.py)."""
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from omg.grpo.paths import references_dir

MMD_REFERENCE_NAME = "mp20_comp_reference.pt"   # torch.save embeds the file stem; keep this name
CREATIVITY_REFERENCE_NAME = "mp20_train_ref.json.gz"

# Source of the creativity reference: the MP-20 training split as distributed with DiffCSP,
# byte-identical to the copy used for the paper.
MP20_TRAIN_CSV_URL = ("https://raw.githubusercontent.com/jiaor17/DiffCSP/"
                      "7121d159826efa2ba9500bf299250d96da37f146/data/mp_20/train.csv")
MP20_TRAIN_CSV_SHA256 = "9c5a34b3684ff4b86d87a86c1c42298d1d28322ea0f03059d9e80ab6a2b729a7"


def default_mmd_reference() -> Path:
    return references_dir() / MMD_REFERENCE_NAME


def default_creativity_reference() -> Path:
    return references_dir() / CREATIVITY_REFERENCE_NAME


def composition_matrix_from_lmdb(lmdb_path, dim: int) -> np.ndarray:
    """[N, dim] fractional-element matrix M[i, Z] = count(Z in structure i) / n_atoms_i, read
    directly from an OMatG LMDB (records store real atomic numbers under "atomic_numbers")."""
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    rows = []
    with env.begin() as txn:
        for _, raw in txn.cursor():
            d = pickle.loads(raw)
            Z = np.asarray(d["atomic_numbers"]).astype(int).ravel()
            assert Z.size > 0, "empty structure in reference lmdb"
            assert Z.min() > 0, "Z=0 found in reference lmdb (unexpected mask token)"
            rows.append(np.bincount(Z, minlength=dim).astype(np.float64)[:dim] / Z.size)
    env.close()
    return np.stack(rows, axis=0)


def creativity_reference_from_csv(csv_path) -> Dict[str, list]:
    """dict[reduced_formula -> list[Structure]] from the MP-20 training CSV (column "cif")."""
    import pandas as pd
    from pymatgen.core import Structure
    ref = defaultdict(list)
    for cif in pd.read_csv(csv_path)["cif"]:
        try:
            s = Structure.from_str(cif, fmt="cif")
            ref[s.composition.reduced_formula].append(s)
        except Exception:
            pass
    return dict(ref)


def save_creativity_reference(ref: Dict[str, list], path) -> None:
    """JSON (pymatgen as_dict) in gzip with a fixed header timestamp, so the file is reproducible."""
    blob = {k: [s.as_dict() for s in v] for k, v in ref.items()}
    raw = json.dumps(blob).encode()
    with open(path, "wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(raw)


def load_creativity_reference(path) -> Dict[str, List]:
    """Load a creativity reference: .json.gz (released format) or .pkl (the format of the runs
    in the paper; both give bit-identical Structures)."""
    from pymatgen.core import Structure
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    with gzip.open(path, "rt") as f:
        blob = json.load(f)
    return {k: [Structure.from_dict(d) for d in v] for k, v in blob.items()}
