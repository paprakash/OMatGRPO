"""Report a LeMat-GenBench result: counts with rates over valid and over the nominal number of
generated structures, the check that all three potentials of the ensemble scored, and (with
--cifs and --summary) the decomposition of mSUN into single-element and sparse-hull structures.

Rates: unique, novel, stable, metastable, SUN and mSUN are given over the valid structures (as
LeMat-GenBench reports them) and over the nominal draw (2,500 in the paper). LeMat-GenBench's
mSUN excludes structures that are on the hull (they count as SUN).

Decomposition join. LeMat-GenBench lists mSUN structures by their position in its valid subset and
records no file names. It loads CIFs in Path.rglob("*.cif") order and drops files pymatgen cannot
parse. The script rebuilds that sequence and proves the mapping element-wise against LeMat-GenBench's
per-structure HHI-production values before reporting; on any disagreement it refuses. Sparse-hull
membership comes from structures_summary.csv (export_cifs.py), joined by file name.

Usage:
  LEMAT_GENBENCH_ROOT=... python scripts/eval/lgb_report.py --result <json> [--nominal 2500]
      [--cifs <dir> --summary <structures_summary.csv> [--msun_list out.csv]]
  (--name <run name> picks the newest results_final/<name>_comprehensive_multi_mlip_hull_*.json)
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

PRESET = "comprehensive_multi_mlip_hull"
FAMILY_ORDER = ["validity", "distribution", "diversity", "novelty", "uniqueness", "hhi", "sun", "stability"]


def lgb_root() -> str:
    root = os.environ.get("LEMAT_GENBENCH_ROOT")
    if not root:
        sys.exit("set LEMAT_GENBENCH_ROOT to the lemat-genbench checkout")
    sys.path.insert(0, f"{root}/scripts")
    sys.path.insert(0, f"{root}/src")
    return root


def newest_result(root, name):
    hits = sorted(glob.glob(f"{root}/results_final/{name}_{PRESET}_*.json"))
    return hits[-1] if hits else None


def three_potentials_intact(stab):
    legs = {m: stab.get(f"stability_n_valid_structures_{m}") for m in ("orb", "mace", "uma")}
    ok = len({int(v) for v in legs.values() if v is not None}) == 1 and legs["uma"] not in (None, 0, 0.0)
    return ok, legs


def rebuild_valid_sequence(cif_dir, valid_ids):
    """Reproduce LeMat-GenBench's load order (unparseable files dropped), then its valid subset."""
    from pymatgen.core import Structure
    loaded = []
    for p in (str(q) for q in Path(cif_dir).rglob("*.cif")):      # the same call LeMat-GenBench makes
        try:
            loaded.append((p, Structure.from_file(p)))
        except Exception:
            continue
    out = []
    for sid in valid_ids:        # structure ids are 0-based positions in the loaded list
        i = int(sid)
        if i < 0 or i >= len(loaded):
            return None
        out.append(loaded[i])
    return out


def verify_mapping(valid_seq, lgb_hhi_values):
    """Recompute HHI-production per structure and compare with LeMat-GenBench's list."""
    from lemat_genbench.metrics.hhi_metrics import HHIProductionMetric
    if not lgb_hhi_values or len(lgb_hhi_values) != len(valid_seq):
        return False, f"length mismatch: ours {len(valid_seq)} vs LeMat-GenBench {len(lgb_hhi_values or [])}"
    metric = HHIProductionMetric(scale_to_0_10=True)
    attrs = metric._get_compute_attributes()
    bad = 0
    for (_, s), ref in zip(valid_seq, lgb_hhi_values):
        try:
            got = HHIProductionMetric.compute_structure(s, **attrs)
        except Exception:
            bad += 1
            continue
        if ref is None or got is None or abs(float(got) - float(ref)) > 1e-6:
            bad += 1
    return bad == 0, f"{bad}/{len(valid_seq)} positions disagree"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", default=None, help="LeMat-GenBench result JSON")
    ap.add_argument("--name", default=None, help="run name; picks the newest result JSON of that name")
    ap.add_argument("--nominal", type=int, default=2500, help="number of generated structures")
    ap.add_argument("--cifs", default=None, help="the CIF folder given to LeMat-GenBench")
    ap.add_argument("--summary", default=None, help="structures_summary.csv from export_cifs.py")
    ap.add_argument("--msun_list", default=None, help="write the per-structure mSUN list to this CSV")
    ap.add_argument("--verbose", action="store_true", help="print every metric LeMat-GenBench emits")
    a = ap.parse_args()

    root = lgb_root()
    from extract_benchmark_metrics import parse_benchmark_string
    path = a.result or (newest_result(root, a.name) if a.name else None)
    if not path:
        sys.exit("no result file (give --result, or --name with results in $LEMAT_GENBENCH_ROOT/results_final)")
    d = json.load(open(path))
    ri, vf = d["run_info"], d["validity_filtering"]
    fams = {f: parse_benchmark_string(d["results"][f]) for f in FAMILY_ORDER if f in d["results"]}
    print(f"result: {path}\nrun_name={ri['run_name']} config={ri['config_name']} timestamp={ri['timestamp']}")
    if a.verbose:
        for fam, fs in fams.items():
            print(f"\n----- {fam} -----")
            for k, v in fs.items():
                print(f"  {k} = " + (f"<per-structure list, len={len(v)}>" if isinstance(v, list) else f"{v}"))

    ok, legs = three_potentials_intact(fams["stability"])
    print(f"\nthree-potential check: orb={legs['orb']} mace={legs['mace']} uma={legs['uma']} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("The ORB/MACE/UMA ensemble is not intact (usually: no Hugging Face token for UMA). "
                 "These numbers are not the paper's setup; fix the token and rerun.")

    sun, uniq, nov = fams["sun"], fams["uniqueness"], fams["novelty"]
    n_scored, n_valid = ri["n_structures"], vf["valid_structures"]
    rows = [("valid", n_valid, n_scored, "scored"),
            ("unique", uniq["unique_structures_count"], n_valid, "valid"),
            ("novel", nov["novel_structures_count"], n_valid, "valid"),
            ("stable", sun["stable_count"], n_valid, "valid"),
            ("metastable", sun["metastable_count"], n_valid, "valid"),
            ("SUN", sun["sun_count"], n_valid, "valid"),
            ("mSUN", sun["msun_count"], n_valid, "valid")]
    print(f"\nnominal={a.nominal} scored={n_scored} valid={n_valid}")
    print(f"| metric | count | rate (denominator) | rate over nominal {a.nominal} |")
    print("|---|---|---|---|")
    for name, cnt, den, denname in rows:
        print(f"| {name} | {cnt} | {cnt / den:.4f} (over {denname} = {den}) | {cnt / a.nominal:.4f} |")
    stab = fams["stability"]
    print(f"\nmean E_hull {stab['stability_mean_e_above_hull']:.4f} eV/atom, mean formation energy "
          f"{stab['mean_formation_energy']:.4f} eV/atom, mean relaxation RMSE {stab['mean_relaxation_RMSE']:.4f} A, "
          f"mean ensemble std {stab['stability_mean_ensemble_std']:.4f}")

    if not (a.cifs and a.summary):
        return
    valid_seq = rebuild_valid_sequence(a.cifs, vf["valid_structure_ids"])
    if valid_seq is None:
        sys.exit("REFUSING: a structure id is out of range of the rebuilt load list")
    ok, detail = verify_mapping(valid_seq, fams["hhi"].get("hhi_production_individual_hhi_values"))
    print(f"\njoin proof (recomputed HHI vs LeMat-GenBench's per-structure list): {'PASS' if ok else 'FAIL'} ({detail})")
    if not ok:
        sys.exit("REFUSING to decompose mSUN on an unverified join")
    ours = {r["cif"]: r for r in csv.DictReader(open(a.summary))}
    msun_idx = sun.get("msun_indices") or []
    arity, single, sparse, unmatched = Counter(), 0, 0, 0
    listing = []
    for rank, i in enumerate(msun_idx):
        p, s = valid_seq[int(i)]
        n_elem = len(s.composition.elements)
        arity[n_elem] += 1
        single += n_elem == 1
        row = ours.get(Path(p).name)
        if row is None:
            unmatched += 1
        elif str(row.get("sparse")).strip().lower() in ("true", "1"):
            sparse += 1
        listing.append([rank, int(i), Path(p).name, s.composition.reduced_formula, n_elem,
                        (row or {}).get("eh_rel", ""), (row or {}).get("ref_rel", ""), (row or {}).get("sparse", "")])
    n = len(msun_idx)
    print(f"\nmSUN = {n}  ({n / a.nominal:.1%} of {a.nominal})")
    print(f"single-element in mSUN: {single} ({single / max(n, 1):.1%})")
    print(f"sparse hull (fewer than 12 reference entries) in mSUN: {sparse} ({sparse / max(n, 1):.1%})")
    print(f"compounds in mSUN: {n - single}")
    print("elements per structure in mSUN: " + ", ".join(f"{k}: {arity[k]}" for k in sorted(arity)))
    if unmatched:
        print(f"[{unmatched} mSUN structures have no row in the summary CSV]")
    if a.msun_list:
        with open(a.msun_list, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["msun_rank", "structure_id", "cif", "formula", "n_elements", "eh_rel", "ref_rel", "sparse"])
            w.writerows(listing)
        print(f"wrote {a.msun_list}")


if __name__ == "__main__":
    main()
