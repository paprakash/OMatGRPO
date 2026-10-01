"""Generate structures from a checkpoint with the paper's evaluation protocol, relax them, apply
the guards and score E_hull.

Protocol (unchanged from the paper):
  - consistent sampler: the 64-point time grid and species noise eta = 0 used in training;
  - N structures in chunks; chunk c is generated after torch.manual_seed(seed * 1000 + c) from
    the c-th batch of the MP-20 validation set (used for the number of atoms per structure only);
  - cell and masked-species guards before relaxation;
  - FIRE relaxation with the cell (Frechet filter), force tolerance 0.05 eV/A, at most 500 steps,
    UMA uma-s-1p2 (task omat);
  - the cell guard again after relaxation;
  - E_hull of the relaxed structure against the LeMat-Bulk UMA hull, with the hull reference count.

Outputs in --out_dir: structures.csv (one row per generated structure), gen.extxyz (generated
structures that passed the first guard), relaxed.extxyz (relaxed structures that passed both
guards), meta.json. A run can be resumed: completed chunks are skipped.

Usage:
  python scripts/eval/generate.py --checkpoint <weights> --out_dir <dir> [--n 2500 --chunk 100 --seed 42]

B200 kernels are not bit-reproducible: two runs give statistically equivalent, not identical,
structures. --deterministic (torch deterministic algorithms) makes generation reproducible, but
its numbers are not those of the paper.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FIELDS = [
    "idx", "chunk", "formula", "n_atoms", "n_elem", "guard_pre", "guard_post",
    "e_unrel", "eh_unrel", "ref_unrel", "e_rel", "eh_rel", "ref_rel", "sparse", "deep_below",
    "rmsd_frac_cart", "vol_gen", "vol_rel", "cell_dF",
]


def guard_classify(pos_np, cell_np, species_np):
    """'masked', 'nan', or the cell-guard bucket ('small', 'oblate', 'huge', 'ok')."""
    from omg.grpo.guards import classify_cell
    if (species_np == 0).any():
        return "masked"
    if not (np.isfinite(pos_np).all() and np.isfinite(cell_np).all()):
        return "nan"
    return classify_cell(cell_np)


def hull_lookup(e_total, symbols):
    """(e_hull, ref_count) against the UMA hull; (nan, 0) if the lookup fails."""
    from pymatgen.core import Composition
    from omg.grpo.lemat_hull import get_energy_above_hull
    try:
        eh, rc = get_energy_above_hull(e_total, Composition(Counter(symbols)), hull_type="uma",
                                       threshold=0.001, return_ref_count=True)
        return float(eh), int(rc)
    except Exception:
        return float("nan"), 0


def build_generator(model_config, checkpoint, data_dir, out_dir, device):
    """Model, interpolants, sampler and validation loader set up for the consistent sampler."""
    from omg.datamodule.dataloader import OMGDataModule
    from omg.grpo.checkpoints import load_policy_weights
    from omg.grpo.train import _lmdb_overlay
    from omg.omg_cli import OMGCLI
    from omg.omg_lightning import OMGLightning
    from omg.omg_trainer import OMGTrainer
    overlay = _lmdb_overlay({"data_dir": str(data_dir)}, Path(out_dir))
    cli = OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
                 run=False, args=[f"--config={model_config}", f"--config={overlay}"])
    lm, dm = cli.model, cli.datamodule
    si, sampler, model = lm.si, lm.sampler, lm.model
    native_T = int(si._integration_time_steps)
    f2s = {df.name: s for df, s in zip(si._data_fields, si._stochastic_interpolants)}
    si._integration_time_steps = 64            # consistent sampler: the training grid
    if "species" in f2s and hasattr(f2s["species"], "_noise"):
        f2s["species"]._noise = 0.0            # and the training species noise
    load_policy_weights(model, checkpoint)
    model = model.to(device).eval()
    return si, sampler, model, dm, native_T


def generate_chunk(si, sampler, model, x1, seed, chunk_idx, device):
    """One chunk of the consistent sampler (exactly the generation of the paper protocol)."""
    torch.manual_seed(seed * 1000 + chunk_idx)
    with torch.no_grad():
        x_0 = sampler.sample_p_0(x1).to(device)
        return si.integrate_consistent(x_0, model, fields=("pos", "cell", "species"), stochastic=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="policy weights (.safetensors, .ckpt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_config", default=str(REPO / "configs" / "prior" / "train.yaml"))
    ap.add_argument("--data_dir", default=None, help="default: OMATGRPO_DATA_DIR or omg/data")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relax_max_steps", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true",
                    help="torch deterministic algorithms (reproducible, but not the paper's numbers)")
    args = ap.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import write as ase_write
    import torch_sim as ts
    from torch_sim.optimizers.cell_filters import CellFilter
    from omg.grpo import paths
    from omg.grpo.guards import EHULL_MAG_FLOOR, EHULL_MIN_REFSET
    from omg.grpo.reward import _DetachedFairChem

    if args.data_dir:
        paths.set_data_dir(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, gen_xyz, rel_xyz = out / "structures.csv", out / "gen.extxyz", out / "relaxed.extxyz"

    done_chunks = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done_chunks = {int(r["chunk"]) for r in csv.DictReader(f)}
        print(f"resume: chunks already done: {sorted(done_chunks)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    si, sampler, model, dm, native_T = build_generator(args.model_config, args.checkpoint,
                                                       paths.data_dir(), out, dev)
    dm.kwargs.update(batch_size=args.chunk, num_workers=0, persistent_workers=False)
    dm.setup("fit")
    val_iter = iter(dm.val_dataloader())

    # UMA with stress for the cell relaxation; outputs detached (see _DetachedFairChem).
    uma = _DetachedFairChem(model="uma-s-1p2", task_name="omat", compute_stress=True)

    write_header = not csv_path.exists()
    n_chunks = (args.n + args.chunk - 1) // args.chunk
    n_lookup_fail = 0
    t_run0 = time.time()
    for ci in range(n_chunks):
        try:
            x1 = next(val_iter)
        except StopIteration:
            val_iter = iter(dm.val_dataloader())
            x1 = next(val_iter)
        if ci in done_chunks:
            continue
        x1 = x1.to(dev)
        t0 = time.time()
        gen = generate_chunk(si, sampler, model, x1, args.seed, ci, dev)
        t_gen = time.time() - t0

        gen_cpu = gen.to("cpu")
        na = gen_cpu.n_atoms.numpy()
        ptr = np.concatenate([[0], np.cumsum(na)])
        pos_all = gen_cpu.pos.float().numpy()          # fractional coordinates
        sp_all = gen_cpu.species.numpy().astype(np.int64)
        cell_all = gen_cpu.cell.float().numpy()

        n_here = len(na)
        atoms_gen, guards = [], []
        for i in range(n_here):
            p, z, c = pos_all[ptr[i]:ptr[i + 1]], sp_all[ptr[i]:ptr[i + 1]], cell_all[i]
            g = guard_classify(p, c, z)
            guards.append(g)
            if g == "ok":
                a = Atoms(numbers=z, scaled_positions=p % 1.0, cell=c, pbc=True)
                a.info["idx"] = ci * args.chunk + i
                atoms_gen.append(a)
            else:
                atoms_gen.append(None)
        ok_idx = [i for i, a in enumerate(atoms_gen) if a is not None]
        ok_atoms = [atoms_gen[i] for i in ok_idx]

        e_unrel = [float("nan")] * n_here
        if ok_atoms:
            res = ts.static(system=ok_atoms, model=uma)
            for j, i in enumerate(ok_idx):
                e_unrel[i] = float(res[j]["potential_energy"].detach().item())

        t1 = time.time()
        rel_atoms = [None] * n_here
        if ok_atoms:
            try:
                st = ts.optimize(
                    system=ok_atoms, model=uma, optimizer=ts.Optimizer.fire,
                    convergence_fn=ts.generate_force_convergence_fn(force_tol=0.05),
                    max_steps=args.relax_max_steps,
                    init_kwargs={"cell_filter": CellFilter.frechet},
                )
                rel = st.to_atoms()
                rel = rel if isinstance(rel, list) else [rel]
                for j, i in enumerate(ok_idx):
                    rel_atoms[i] = rel[j]
                del st
            except (RuntimeError, TimeoutError) as e:
                print(f"[chunk {ci}] relaxation failed ({e}); relaxed slots stay empty")
        gc.collect()
        torch.cuda.empty_cache()
        t_rel = time.time() - t1

        guard_post = ["na"] * n_here
        e_rel = [float("nan")] * n_here
        rel_ok_idx = []
        for i in ok_idx:
            a = rel_atoms[i]
            if a is None:
                guard_post[i] = "relax_fail"
                continue
            g = guard_classify(a.get_positions(), np.array(a.get_cell()), a.get_atomic_numbers())
            guard_post[i] = g
            if g == "ok":
                rel_ok_idx.append(i)
        if rel_ok_idx:
            res = ts.static(system=[rel_atoms[i] for i in rel_ok_idx], model=uma)
            for j, i in enumerate(rel_ok_idx):
                e_rel[i] = float(res[j]["potential_energy"].detach().item())
        gc.collect()
        torch.cuda.empty_cache()

        for i in rel_ok_idx:              # relaxed frames keep the index of their generated structure
            rel_atoms[i].info["idx"] = ci * args.chunk + i

        rows = []
        for i in range(n_here):
            syms = [chemical_symbols[int(v)] for v in sp_all[ptr[i]:ptr[i + 1]] if v > 0]
            eh_u, rc_u = hull_lookup(e_unrel[i], syms) if np.isfinite(e_unrel[i]) else (float("nan"), 0)
            eh_r, rc_r = hull_lookup(e_rel[i], syms) if np.isfinite(e_rel[i]) else (float("nan"), 0)
            n_lookup_fail += int(np.isfinite(e_rel[i]) and not np.isfinite(eh_r))
            sparse = bool(rc_r < EHULL_MIN_REFSET) if np.isfinite(eh_r) else True
            deep_below = bool(np.isfinite(eh_r) and eh_r < EHULL_MAG_FLOOR)
            rmsd = float("nan")
            vol_g = float(abs(np.linalg.det(cell_all[i])))
            vol_r, cell_dF = float("nan"), float("nan")
            a_rel = rel_atoms[i]
            if a_rel is not None and guards[i] == "ok":
                c_g, c_r = cell_all[i], np.array(a_rel.get_cell())
                vol_r = float(abs(np.linalg.det(c_r)))
                cell_dF = float(np.linalg.norm(c_r - c_g))
                try:   # fractional displacement, mapped through the generated cell
                    df = a_rel.get_scaled_positions() - pos_all[ptr[i]:ptr[i + 1]] % 1.0
                    df -= np.round(df)
                    rmsd = float(np.sqrt(((df @ c_g) ** 2).sum(axis=1).mean()))
                except Exception:
                    pass
            cc = Counter(syms)
            rows.append(dict(
                idx=ci * args.chunk + i, chunk=ci, formula="".join(f"{s}{cc[s]}" for s in sorted(cc)),
                n_atoms=int(na[i]), n_elem=len(cc), guard_pre=guards[i], guard_post=guard_post[i],
                e_unrel=e_unrel[i], eh_unrel=eh_u, ref_unrel=rc_u, e_rel=e_rel[i], eh_rel=eh_r,
                ref_rel=rc_r, sparse=sparse, deep_below=deep_below, rmsd_frac_cart=rmsd,
                vol_gen=vol_g, vol_rel=vol_r, cell_dF=cell_dF,
            ))
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerows(rows)
        ase_write(str(gen_xyz), [a for a in atoms_gen if a is not None], format="extxyz", append=True)
        ase_write(str(rel_xyz), [rel_atoms[i] for i in rel_ok_idx], format="extxyz", append=True)

        eh = np.array([r["eh_rel"] for r in rows], float)
        fin = np.isfinite(eh)
        print(f"[chunk {ci + 1}/{n_chunks}] n={n_here} ok={len(ok_idx)} relaxed_ok={len(rel_ok_idx)} "
              f"eh_rel_median={np.median(eh[fin]) if fin.any() else float('nan'):.3f} "
              f"sparse={sum(r['sparse'] for r in rows)}/{n_here} gen={t_gen:.0f}s relax={t_rel:.0f}s",
              flush=True)

    from omg.grpo.train import _provenance
    meta = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=_sha256(args.checkpoint),
                model_config=args.model_config, n=args.n, chunk=args.chunk, seed=args.seed,
                sampler="consistent", time_grid=int(si._integration_time_steps), native_time_grid=native_T,
                species_eta=0.0, relax=f"FIRE, Frechet cell filter, force_tol 0.05, max {args.relax_max_steps} steps",
                potential="uma-s-1p2 (omat)", hull="LeMat-Bulk-MLIP-Hull uma, threshold 0.001",
                deterministic=args.deterministic, hull_lookup_failures=n_lookup_fail,
                wall_s=round(time.time() - t_run0, 1), **_provenance())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
