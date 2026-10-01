"""Export the valid relaxed structures of a generate.py run as CIFs for LeMat-GenBench.

A structure is valid when it passed the cell and masked-species guards before and after the
relaxation and has a finite relaxed E_hull. Only valid structures are exported, as
<dest>/cifs/{idx:04d}_{formula}.cif. <dest>/structures_summary.csv has one row per generated
structure (idx, formula, n_elements, eh_rel, ref_rel, valid, trusted, sparse, deep_below, cif),
plus the novelty columns when novelty.py has run (novelty.csv in the run folder).

The CIF writer is pymatgen's Structure.to(filename=...), as for the paper's sets: LeMat-GenBench's
pinned pymatgen rejects some CIFs, so another writer could change the number of scored structures.

Usage: python scripts/eval/export_cifs.py --run_dir <generate.py out_dir> --dest <dir>
"""
import argparse
import csv
import math
import re
from pathlib import Path


def is_valid(row) -> bool:
    eh = float(row["eh_rel"]) if row["eh_rel"] not in ("", "nan") else float("nan")
    return row["guard_pre"] == "ok" and row["guard_post"] == "ok" and math.isfinite(eh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    run, dest = Path(args.run_dir), Path(args.dest)
    cif_dir = dest / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    rows = {int(r["idx"]): r for r in csv.DictReader(open(run / "structures.csv"))}
    novelty = {}
    if (run / "novelty.csv").exists():
        novelty = {int(r["idx"]): r for r in csv.DictReader(open(run / "novelty.csv"))}
    frames = {int(a.info["idx"]): a for a in read(str(run / "relaxed.extxyz"), index=":")}

    out_rows, n_written = [], 0
    for idx in sorted(rows):
        r = rows[idx]
        valid = is_valid(r)
        sparse, deep = r["sparse"] == "True", r["deep_below"] == "True"
        cif_name = ""
        if valid and idx in frames:
            s = AseAtomsAdaptor.get_structure(frames[idx])
            cif_name = f"{idx:04d}_{r['formula']}.cif"
            s.to(filename=str(cif_dir / cif_name))
            n_written += 1
        out = dict(idx=idx, formula=r["formula"], n_elements=len(set(re.findall(r"([A-Z][a-z]?)\d", r["formula"]))),
                   eh_rel=r["eh_rel"], ref_rel=r["ref_rel"], valid=valid,
                   trusted=valid and not sparse and not deep, sparse=r["sparse"], deep_below=r["deep_below"])
        if novelty:
            n = novelty.get(idx, {})
            for key in ("unique", "novel_mp20", "msun_mp20", "novel_alexmp", "msun_alexmp"):
                if key in n:
                    out[key] = n[key]
        out["cif"] = cif_name
        out_rows.append(out)

    with open(dest / "structures_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{dest}: {n_written} CIFs written, {len(out_rows)} rows "
          f"({sum(1 for r in out_rows if r['trusted'])} trusted)")


if __name__ == "__main__":
    main()
