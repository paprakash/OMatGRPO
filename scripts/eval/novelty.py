"""Optional internal evaluation stage: uniqueness, novelty and mSUN of a generate.py run.

Definitions (pymatgen StructureMatcher defaults: ltol 0.2, stol 0.3, angle_tol 5):
  valid     both guards passed and the relaxed E_hull is finite
  stable01  valid and E_hull <= 0.1 eV/atom (metastable); stable00: E_hull <= 0
  unique    first representative of its structure-match cluster within the run
  novel     no structure match among the reference structures with the same reduced formula
  msun      stable01 and unique and novel, counted over ALL generated structures
References: MP-20 train (the creativity reference of training, so this novelty and the creativity
term share one definition), and optionally Alex-MP-20 (--alexmp_csv, the MatterGen data release).

This is not the LeMat-GenBench evaluation of the paper's tables (see run_lemat_genbench.sh).
Writes <run_dir>/novelty.csv and <run_dir>/novelty_summary.json.

Usage: python scripts/eval/novelty.py --run_dir <dir> [--alexmp_csv train.csv val.csv --alexmp_cache ref.json.gz]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - hw, c + hw


def load_alexmp(csvs, cache):
    from omg.grpo import references as refs
    if cache and Path(cache).exists():
        return refs.load_creativity_reference(cache)
    ref = defaultdict(list)
    for path in csvs:
        part = refs.creativity_reference_from_csv(path)
        for k, v in part.items():
            ref[k].extend(v)
    ref = dict(ref)
    if cache:
        refs.save_creativity_reference(ref, cache)
    return ref


def novel_flags(structs, idxs_by_formula, reference, matcher):
    flags = {}
    for formula, idxs in idxs_by_formula.items():
        cands = reference.get(formula, [])
        for i in idxs:
            flags[i] = not any(matcher.fit(structs[i], t) for t in cands)
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mp20_reference", default=None,
                    help="default: <data_dir>/references/mp20_train_ref.json.gz")
    ap.add_argument("--alexmp_csv", nargs="*", default=None, help="Alex-MP-20 CSV files with a 'cif' column")
    ap.add_argument("--alexmp_cache", default=None, help="json.gz cache of the parsed Alex-MP-20 reference")
    args = ap.parse_args()

    from ase.io import read as ase_read
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    from omg.grpo import references as refs

    run = Path(args.run_dir)
    rows = list(csv.DictReader(open(run / "structures.csv")))
    structs = {}
    for a in ase_read(str(run / "relaxed.extxyz"), index=":"):
        try:
            structs[int(a.info["idx"])] = AseAtomsAdaptor.get_structure(a)
        except Exception:
            pass
    for r in rows:
        r["idx"] = int(r["idx"])
        r["eh_rel"] = float(r["eh_rel"]) if r["eh_rel"] not in ("", "nan") else float("nan")
        r["valid"] = r["guard_pre"] == "ok" and r["guard_post"] == "ok" and np.isfinite(r["eh_rel"])

    matcher = StructureMatcher()
    by_formula = defaultdict(list)
    for r in rows:
        if r["valid"] and r["idx"] in structs:
            by_formula[structs[r["idx"]].composition.reduced_formula].append(r["idx"])
    unique = {}
    for idxs in by_formula.values():
        reps = []
        for i in idxs:
            dup = any(matcher.fit(structs[i], structs[j]) for j in reps)
            unique[i] = not dup
            if not dup:
                reps.append(i)

    mp20 = refs.load_creativity_reference(args.mp20_reference or refs.default_creativity_reference())
    novel = {"mp20": novel_flags(structs, by_formula, mp20, matcher)}
    if args.alexmp_csv or args.alexmp_cache:
        novel["alexmp"] = novel_flags(structs, by_formula, load_alexmp(args.alexmp_csv or [], args.alexmp_cache),
                                      matcher)

    out_rows = []
    for r in rows:
        i = r["idx"]
        o = dict(idx=i, valid=r["valid"], stable01=bool(r["valid"] and r["eh_rel"] <= 0.1),
                 stable00=bool(r["valid"] and r["eh_rel"] <= 0.0), unique=bool(unique.get(i, False)))
        for ref_name, flags in novel.items():
            o[f"novel_{ref_name}"] = bool(flags.get(i, False))
            o[f"msun_{ref_name}"] = o["stable01"] and o["unique"] and o[f"novel_{ref_name}"]
            o[f"sun0_{ref_name}"] = o["stable00"] and o["unique"] and o[f"novel_{ref_name}"]
        out_rows.append(o)
    with open(run / "novelty.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    summary = {"n": n}
    for key in [k for k in out_rows[0] if k != "idx"]:
        k = sum(1 for r in out_rows if r[key])
        p, lo, hi = wilson(k, n)
        summary[key] = {"count": k, "rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    (run / "novelty_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
