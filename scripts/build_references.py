"""Build the two MP-20 references of the reward into <data_dir>/references.

  mmd         composition matrix of the MMD bonus, from <data_dir>/mp_20/train.lmdb
              (10,000 rows subsampled with seed 0). Reproduces the file used in the paper
              exactly (sha256 below).
  creativity  structures of the creativity term, bucketed by reduced formula, from the MP-20
              training CSV (downloaded from a pinned DiffCSP commit and checked by sha256, or
              given with --csv). Rebuilt this way, the formulas, the number of structures per
              formula, the species and the fractional coordinates equal the reference used in the
              paper; 2,650 of 27,136 lattice matrices differ by at most 6.2e-15 A (floating-point
              rounding when the lattice is built). The paper's original file is not distributed;
              --from_pkl converts that original pickle if you have it.

Usage:
  python scripts/build_references.py mmd [--data_dir DIR]
  python scripts/build_references.py creativity [--data_dir DIR] [--csv PATH | --from_pkl PATH]
"""
import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omg.grpo import paths  # noqa: E402
from omg.grpo import references as refs  # noqa: E402

MMD_REFERENCE_SHA256 = "a93951dc666a8dc41695068e327ff4960a2a41054a34da808b5e91fbd6b1cab2"
MMD_MAX_REFERENCE = 10000
MMD_SEED = 0


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mmd(out_dir: Path) -> Path:
    import torch
    from omg.grpo.mmd_diversity import COMP_DIM
    lmdb_path = paths.data_dir() / "mp_20" / "train.lmdb"
    if not lmdb_path.exists():
        sys.exit(f"{lmdb_path} not found; run scripts/download_assets.py first.")
    M = refs.composition_matrix_from_lmdb(lmdb_path, COMP_DIM)
    n_total = M.shape[0]
    mat = torch.from_numpy(M)
    if n_total > MMD_MAX_REFERENCE:
        g = torch.Generator().manual_seed(MMD_SEED)
        idx = torch.randperm(n_total, generator=g)[:MMD_MAX_REFERENCE]
        mat = mat[idx].contiguous()
    row_sums = mat.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert float(mat[:, 0].abs().max()) == 0.0
    out = out_dir / refs.MMD_REFERENCE_NAME
    torch.save({"matrix": mat, "dim": COMP_DIM, "n_total": n_total, "seed": MMD_SEED}, out)
    digest = sha256(out)
    status = "matches" if digest == MMD_REFERENCE_SHA256 else "DOES NOT MATCH"
    print(f"{out}: {tuple(mat.shape)} from {n_total} structures; sha256 {digest} "
          f"({status} the reference used in the paper)")
    return out


def build_creativity(out_dir: Path, csv=None, from_pkl=None) -> Path:
    out = out_dir / refs.CREATIVITY_REFERENCE_NAME
    if from_pkl:
        ref = refs.load_creativity_reference(from_pkl)
        source = from_pkl
    else:
        if csv is None:
            tmp = Path(tempfile.mkdtemp()) / "train.csv"
            print(f"downloading {refs.MP20_TRAIN_CSV_URL}")
            urllib.request.urlretrieve(refs.MP20_TRAIN_CSV_URL, tmp)
            csv = tmp
        digest = sha256(csv)
        if digest != refs.MP20_TRAIN_CSV_SHA256:
            sys.exit(f"{csv}: sha256 {digest} != expected {refs.MP20_TRAIN_CSV_SHA256}")
        ref = refs.creativity_reference_from_csv(csv)
        source = csv
    refs.save_creativity_reference(ref, out)
    n = sum(len(v) for v in ref.values())
    print(f"{out}: {n} structures, {len(ref)} reduced formulas (from {source}); sha256 {sha256(out)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["mmd", "creativity", "all"])
    ap.add_argument("--data_dir", default=None, help="data directory (default: OMATGRPO_DATA_DIR or omg/data)")
    ap.add_argument("--csv", default=None, help="local MP-20 train.csv instead of the download")
    ap.add_argument("--from_pkl", default=None, help="convert an existing creativity reference .pkl")
    args = ap.parse_args()
    if args.data_dir:
        paths.set_data_dir(args.data_dir)
    out_dir = paths.references_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.which in ("mmd", "all"):
        build_mmd(out_dir)
    if args.which in ("creativity", "all"):
        build_creativity(out_dir, csv=args.csv, from_pkl=args.from_pkl)


if __name__ == "__main__":
    main()
