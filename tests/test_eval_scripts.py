"""Pieces of the evaluation scripts that run without a GPU, UMA or LeMat-GenBench."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

import export_cifs  # noqa: E402
import generate  # noqa: E402
import lgb_report  # noqa: E402


def test_guard_classify():
    cell = np.diag([4.0, 4.0, 4.0])
    pos = np.zeros((2, 3))
    assert generate.guard_classify(pos, cell, np.array([11, 17])) == "ok"
    assert generate.guard_classify(pos, cell, np.array([0, 17])) == "masked"
    assert generate.guard_classify(pos * np.nan, cell, np.array([11, 17])) == "nan"
    assert generate.guard_classify(pos, np.diag([1.0, 4.0, 4.0]), np.array([11, 17])) == "small"


def test_validity_rule():
    row = {"guard_pre": "ok", "guard_post": "ok", "eh_rel": "0.05"}
    assert export_cifs.is_valid(row)
    assert not export_cifs.is_valid({**row, "eh_rel": "nan"})
    assert not export_cifs.is_valid({**row, "guard_post": "relax_fail"})


def test_export_writes_only_valid_structures(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    fields = ["idx", "formula", "guard_pre", "guard_post", "eh_rel", "ref_rel", "sparse", "deep_below"]
    rows = [dict(idx=0, formula="Cl1Na1", guard_pre="ok", guard_post="ok", eh_rel=0.05, ref_rel=30,
                 sparse=False, deep_below=False),
            dict(idx=1, formula="Mg1O1", guard_pre="ok", guard_post="small", eh_rel="nan", ref_rel=0,
                 sparse=True, deep_below=False)]
    with open(run / "structures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    a = Atoms("NaCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 4.0, pbc=True)
    a.info["idx"] = 0
    write(str(run / "relaxed.extxyz"), [a], format="extxyz")
    subprocess.run([sys.executable, str(REPO / "scripts/eval/export_cifs.py"), "--run_dir", str(run),
                    "--dest", str(tmp_path / "out")], check=True)
    assert sorted(p.name for p in (tmp_path / "out" / "cifs").iterdir()) == ["0000_Cl1Na1.cif"]
    summary = list(csv.DictReader(open(tmp_path / "out" / "structures_summary.csv")))
    assert [r["valid"] for r in summary] == ["True", "False"]
    assert summary[0]["trusted"] == "True" and summary[0]["cif"] == "0000_Cl1Na1.cif"


def test_three_potential_check():
    ok, _ = lgb_report.three_potentials_intact({"stability_n_valid_structures_orb": 1745.0,
                                                "stability_n_valid_structures_mace": 1745.0,
                                                "stability_n_valid_structures_uma": 1745.0})
    assert ok
    ok, _ = lgb_report.three_potentials_intact({"stability_n_valid_structures_orb": 1745.0,
                                                "stability_n_valid_structures_mace": 1745.0,
                                                "stability_n_valid_structures_uma": 0.0})
    assert not ok
