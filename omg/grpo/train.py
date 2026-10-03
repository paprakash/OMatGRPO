"""OMatGRPO training entry point.

    python -m omg.grpo.train --config configs/runs/arityguard_creatrelax.yaml [--flag value ...]

Settings resolve in this order: built-in defaults (the OMatGRPO run), then the config file, then
flags given on the command line. Unknown flags and unknown config keys are errors.
--print_config prints the resolved settings and exits.
"""
import argparse
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from omg.datamodule.dataloader import OMGDataModule
from omg.grpo.checkpoints import load_policy_weights
from omg.grpo.grpo_lightning import OMatGRPOModule
from omg.grpo.paths import set_data_dir
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = "configs/prior/train.yaml"   # relative paths resolve from the cwd, else the repo
PRIOR_CHECKPOINT_NAME = "prior.safetensors"
ALLOWED_FIELDS = ("pos", "pos,cell", "pos,cell,species")

# Values fixed in the code (not settable): PPO clip, gradient-norm clip, precision.
EPS_CLIP = 0.2
GRAD_CLIP_NORM = 0.5          # applied manually in training_step
PRECISION = "bf16-mixed"


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _float_or_inf(value) -> float:
    return float(value)   # accepts "-inf"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omatgrpo-train", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    def flag(group, name, type_, default, help_, choices=None, metavar=None):
        kw = dict(default=default, help=f"{help_} (default: {default})")
        if metavar:
            kw["metavar"] = metavar
        if type_ is _bool:
            kw.update(type=_bool, nargs="?", const=True, metavar="true|false")
        else:
            kw.update(type=type_)
        if choices:
            kw["choices"] = choices
        group.add_argument(f"--{name}", **kw)

    g = p.add_argument_group("run")
    g.add_argument("--config", default=None,
                   help="YAML file with flag values (keys = flag names without --); command-line flags override it")
    g.add_argument("--print_config", action="store_true", help="print the resolved settings and exit")
    flag(g, "run_name", str, "omatgrpo", "name of the run (output folder and wandb run name)")
    flag(g, "output_dir", str, None, "folder for checkpoints, the final model and the resolved config; "
         "None means outputs/<run_name>")
    flag(g, "data_dir", str, None, "data folder (MP-20 LMDBs, references, hull cache, prior); None means "
         "OMATGRPO_DATA_DIR or omg/data")
    flag(g, "model_config", str, DEFAULT_MODEL_CONFIG, "OMatG config of the prior (model, interpolants, data)")
    flag(g, "init_checkpoint", str, None, "weights of the prior; the policy starts from them and they are the KL "
         "reference. None means <data_dir>/prior/" + PRIOR_CHECKPOINT_NAME)
    flag(g, "resume", str, None, "periodic checkpoint of an interrupted run to resume (policy, optimizer, step)")
    flag(g, "seed", int, 0, "global seed (the paper runs used 0; B200 kernels are not bit-reproducible anyway)")
    flag(g, "checkpoint_every", int, 50, "save a periodic checkpoint every N rollouts")
    flag(g, "wandb_project", str, "omatgrpo", "wandb project")
    flag(g, "wandb_mode", str, "offline", "wandb mode: offline (logs stay in <output_dir>/wandb; upload later "
         "with `wandb sync`), online, or disabled", choices=["online", "offline", "disabled"])

    g = p.add_argument_group("rollouts and optimization")
    flag(g, "fields", str, "pos,cell,species", "channels the policy learns: pos, pos,cell or pos,cell,species",
         choices=list(ALLOWED_FIELDS), metavar="FIELDS")
    flag(g, "freeze_composition", _bool, False, "sample one composition per group from the frozen prior and "
         "learn only positions and lattice (the frozen-composition control); needs --fields pos,cell and "
         "--occurrence_discount false")
    flag(g, "num_groups", int, 4, "B, groups per rollout")
    flag(g, "group_size", int, 16, "K, structures per group (members of a group have the same number of atoms)")
    flag(g, "time_grid", int, 64, "points of the integration time grid (steps = points - 1); capped at the "
         "value in the model config")
    flag(g, "species_eta", float, 0.0, "noise eta of the discrete species channel (the closed-form species KL "
         "needs 0)")
    flag(g, "rollouts", int, 750, "number of rollouts (optimizer steps = rollouts x inner_epochs)")
    flag(g, "inner_epochs", int, 3, "PPO epochs per rollout")
    flag(g, "lr", float, 1e-4, "AdamW learning rate")
    flag(g, "alpha_pos", float, 0.1, "weight of the position surrogate")
    flag(g, "alpha_cell", float, 0.1, "weight of the lattice surrogate")
    flag(g, "alpha_species", float, 1.0, "weight of the species surrogate")
    flag(g, "beta_kl_pos", float, 0.01, "weight of the position KL to the prior (0 = off)")
    flag(g, "beta_kl_species", float, 0.05, "weight of the species KL to the prior (0 = off)")

    g = p.add_argument_group("reward: stability term and guards")
    flag(g, "w_stability", float, 1.0, "weight of the stability term -clip(E_hull); 0 turns it off together "
         "with every guard penalty")
    flag(g, "relax", _bool, True, "relax structures (FIRE, UMA) before scoring")
    flag(g, "relax_cell", _bool, True, "include the cell in the relaxation (Frechet filter), with a "
         "post-relaxation cell guard")
    flag(g, "relax_steps", int, 100, "maximum FIRE steps of the relaxation")
    flag(g, "stability_floor_at_zero", _bool, True, "clip E_hull at 0 from below, so depth below the hull "
         "earns nothing")
    flag(g, "stability_cap", float, 1.0, "upper clip of E_hull (eV/atom), also the value of penalized and "
         "failed structures")
    flag(g, "deep_below_hull", _float_or_inf, -0.1, "E_hull below this counts as deep below the hull and is "
         "penalized; --deep_below_hull=-inf turns the guard off")
    flag(g, "sparse_gate", _bool, True, "treat structures on sparse hulls (fewer than --sparse_min_refs "
         "reference entries) as untrusted for either E_hull sign")
    flag(g, "sparse_min_refs", int, 12, "minimum hull reference entries of a trusted E_hull")
    flag(g, "sparse_route", str, "abstain", "what untrusted sparse structures get: abstain (zero advantage) or "
         "penalty (the cap value)", choices=["abstain", "penalty"])
    flag(g, "single_element_guard", str, "on", "penalize single-element structures", choices=["on", "off"])

    g = p.add_argument_group("reward: diversity and creativity")
    flag(g, "occurrence_discount", _bool, True, "pull repeated compositions within a group toward the worst reward")
    flag(g, "occurrence_tol", int, 3, "occurrences in a group with no discount")
    flag(g, "occurrence_zero", int, 6, "occurrences in a group at which the discount is complete")
    flag(g, "w_mmd", float, 0.4, "weight of the compositional MMD coverage bonus (0 = off)")
    flag(g, "mmd_comp_reference", str, None, "MMD composition reference; None means "
         "<data_dir>/references/mp20_comp_reference.pt")
    flag(g, "mmd_kernel", str, "poly", "MMD kernel", choices=["poly", "linear"])
    flag(g, "mmd_poly_c", float, 1.0, "polynomial kernel offset c")
    flag(g, "mmd_poly_d", int, 3, "polynomial kernel degree d")
    flag(g, "mmd_max_reference", int, 10000, "rows of the MMD reference used (seeded subsample)")
    flag(g, "mmd_norm", str, "minmax", "per-batch scaling of the MMD credit", choices=["minmax", "zscore", "none"])
    flag(g, "w_creat", float, 1.0, "weight of the creativity term (unique and novel vs MP-20 train; 0 = off)")
    flag(g, "creat_on_relaxed", _bool, True, "score creativity on the relaxed structures (needs --relax true)")
    flag(g, "creativity_reference", str, None, "creativity reference; None means "
         "<data_dir>/references/mp20_train_ref.json.gz")
    flag(g, "creativity_sm_timeout", float, 3.0, "StructureMatcher timeout per structure (s, rounded to an "
         "integer >= 1); a timeout scores 0")

    g = p.add_argument_group("reward variants of the reward-hacking appendix")
    flag(g, "reward_type", str, "e_hull", "e_hull (paper), or the formation / absolute energy rewards",
         choices=["e_hull", "formation", "absolute"])
    flag(g, "w_rmsd_geom", float, 0.0, "weight of the residual geometry term (0 = off)")
    flag(g, "rmsd_geom_clamp", float, 3.0, "clamp of the residual geometry term (A)")
    flag(g, "w_rmsd", float, 0.0, "weight of the displacement reward (0 = off)")
    flag(g, "fmax", float, 10.0, "force tolerance of the displacement-reward relaxation")
    flag(g, "fmax_schedule", str, None, "step:fmax pairs for the displacement reward, e.g. 0:10,1000:5")
    flag(g, "reward_offset", float, 0.3, "offset of the displacement reward")
    return p


def flag_table_markdown() -> str:
    """Markdown table of every flag, grouped as in --help (used to generate the README section)."""
    parser = build_parser()
    lines = []
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.dest != "help"]
        if not actions:
            continue
        lines += ["", f"**{group.title}**", "", "| flag | default | description |", "|---|---|---|"]
        for a in actions:
            desc = (a.help or "").split(" (default:")[0].replace("|", "\\|")
            if a.choices:
                desc += " (" + ", ".join(str(c) for c in a.choices) + ")"
            default = "" if isinstance(a, argparse._StoreTrueAction) else f"`{a.default}`"
            lines.append(f"| `--{a.dest}` | {default} | {desc} |")
    return "\n".join(lines).strip() + "\n"


def _parser_dests(parser) -> List[str]:
    return [a.dest for a in parser._actions if a.dest not in ("help",)]


def resolve_config(argv: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Resolve the settings from defaults, an optional --config file and command-line flags.

    Pure apart from reading the config file: `env` (default os.environ) is only used for the data
    directory. Raises SystemExit on unknown flags and ValueError on unknown config keys or invalid
    combinations.
    """
    import os
    import yaml
    env = os.environ if env is None else env
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    file_values: Dict[str, Any] = {}
    if known.config:
        with open(known.config) as f:
            file_values = yaml.safe_load(f) or {}
        dests = set(_parser_dests(parser)) - {"config", "print_config"}
        unknown = sorted(set(file_values) - dests)
        if unknown:
            raise ValueError(f"{known.config}: unknown keys {unknown}")
        for action in parser._actions:
            if action.dest in file_values and action.choices is not None:
                if file_values[action.dest] not in action.choices:
                    raise ValueError(f"{known.config}: {action.dest}={file_values[action.dest]!r} not in "
                                     f"{list(action.choices)}")
            if action.dest in file_values and action.type is _bool:
                file_values[action.dest] = _bool(file_values[action.dest])
        parser.set_defaults(**file_values)
    cfg = vars(parser.parse_args(argv))

    data_dir = Path(cfg["data_dir"] or env.get("OMATGRPO_DATA_DIR") or (REPO_ROOT / "omg" / "data"))
    data_dir = data_dir.expanduser().resolve()
    cfg["data_dir"] = str(data_dir)
    model_config = Path(cfg["model_config"])
    if not model_config.is_absolute() and not model_config.exists():
        model_config = REPO_ROOT / model_config
    cfg["model_config"] = str(model_config)
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(Path("outputs") / cfg["run_name"])
    if cfg["init_checkpoint"] is None:
        cfg["init_checkpoint"] = str(data_dir / "prior" / PRIOR_CHECKPOINT_NAME)
    if cfg["mmd_comp_reference"] is None and cfg["w_mmd"] > 0:
        cfg["mmd_comp_reference"] = str(data_dir / "references" / "mp20_comp_reference.pt")
    if cfg["creativity_reference"] is None and cfg["w_creat"] > 0:
        cfg["creativity_reference"] = str(data_dir / "references" / "mp20_train_ref.json.gz")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Reject combinations that cannot work or that silently do something else than intended."""
    fields = cfg["fields"].split(",")
    errors = []
    if cfg["fields"] not in ALLOWED_FIELDS:
        errors.append(f"--fields must be one of {list(ALLOWED_FIELDS)}")
    if cfg["freeze_composition"]:
        if "species" in fields:
            errors.append("--freeze_composition true needs --fields pos,cell (the composition is fixed)")
        if cfg["occurrence_discount"]:
            errors.append("--freeze_composition true needs --occurrence_discount false: all members of a "
                          "group share one composition, so the discount would set every reward to the "
                          "floor and every group would be dead")
    if cfg["creat_on_relaxed"] and cfg["w_creat"] > 0 and not cfg["relax"]:
        errors.append("--creat_on_relaxed true needs --relax true; set --creat_on_relaxed false to score "
                      "creativity on unrelaxed structures")
    if cfg["creat_on_relaxed"] and cfg["w_creat"] > 0 and cfg["reward_type"] != "e_hull":
        errors.append("--creat_on_relaxed true needs --reward_type e_hull")
    if cfg["stability_cap"] <= 0:
        errors.append("--stability_cap must be > 0")
    if not cfg["occurrence_tol"] < cfg["occurrence_zero"]:
        errors.append("--occurrence_tol must be smaller than --occurrence_zero")
    if cfg["group_size"] < 2:
        errors.append("--group_size must be >= 2 (advantages are relative within a group)")
    for name in ("num_groups", "rollouts", "inner_epochs", "relax_steps", "checkpoint_every"):
        if cfg[name] < 1:
            errors.append(f"--{name} must be >= 1")
    if cfg["time_grid"] < 2:
        errors.append("--time_grid must be >= 2")
    for name in ("w_mmd", "w_creat", "w_stability", "beta_kl_pos", "beta_kl_species", "w_rmsd", "w_rmsd_geom"):
        if cfg[name] < 0:
            errors.append(f"--{name} must be >= 0")
    if "species" in fields and cfg["beta_kl_species"] > 0 and cfg["species_eta"] != 0.0:
        errors.append("--beta_kl_species > 0 needs --species_eta 0 (the closed-form species KL assumes it)")
    if math.isnan(cfg["deep_below_hull"]):
        errors.append("--deep_below_hull must be a number or -inf")
    if errors:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(errors))
    if "species" not in fields and cfg["beta_kl_species"] > 0:
        warnings.warn("--beta_kl_species has no effect because species is not learned")


def module_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments of OMatGRPOModule (without si, sampler and model) for a resolved config."""
    fmax_schedule = None
    if cfg["fmax_schedule"]:
        fmax_schedule = [(int(s), float(v)) for s, v in
                         (pair.split(":") for pair in cfg["fmax_schedule"].split(","))]
    reward_cfg = {
        "fmax": cfg["fmax"],
        "weights": {"rmsd": cfg["w_rmsd"], "energy": 1.0},
        "reward_type": cfg["reward_type"],
        "relax_before_reward": cfg["relax"],
        "relax_max_steps": cfg["relax_steps"],
        "w_rmsd_geom": cfg["w_rmsd_geom"],
        "rmsd_geom_clamp": cfg["rmsd_geom_clamp"],
        "w_ehull": cfg["w_stability"],
        "ehull_mag_floor": cfg["deep_below_hull"],
        "ehull_min_refset": cfg["sparse_min_refs"],
        "ehull_sparse_gate": cfg["sparse_gate"],
        "sparse_route": "neutral" if cfg["sparse_route"] == "abstain" else "worst",
        "route_elemental": "worst" if cfg["single_element_guard"] == "on" else "off",
        "ehull_floor_at_zero": cfg["stability_floor_at_zero"],
        "ehull_cap": cfg["stability_cap"],
        "relax_cell_dof": cfg["relax_cell"],
        "w_creat": cfg["w_creat"],
        "creativity_reference": cfg["creativity_reference"],
        "creativity_sm_timeout": cfg["creativity_sm_timeout"],
        "creat_on_relaxed": cfg["creat_on_relaxed"],
    }
    if fmax_schedule is not None:
        reward_cfg["fmax_schedule"] = fmax_schedule
    use_mmd = cfg["w_mmd"] > 0
    return {
        "k": cfg["group_size"],
        "eps_clip": EPS_CLIP,
        "beta_kl": cfg["beta_kl_pos"],
        "beta_kl_species": cfg["beta_kl_species"],
        "lr": cfg["lr"],
        "fields": tuple(cfg["fields"].split(",")),
        "num_inner_epochs": cfg["inner_epochs"],
        "reward_offset": cfg["reward_offset"],
        "reward_cfg": reward_cfg,
        "dng_mode": cfg["freeze_composition"],
        "alpha_pos": cfg["alpha_pos"],
        "alpha_cell": cfg["alpha_cell"],
        "alpha_species": cfg["alpha_species"],
        "use_diversity": cfg["occurrence_discount"],
        "div_tol": cfg["occurrence_tol"],
        "div_buff": cfg["occurrence_zero"],
        "use_mmd_diversity": use_mmd,
        "w_mmd_diversity": cfg["w_mmd"],
        "mmd_comp_reference": cfg["mmd_comp_reference"] if use_mmd else None,
        "mmd_kernel": cfg["mmd_kernel"],
        "mmd_poly_c": cfg["mmd_poly_c"],
        "mmd_poly_d": cfg["mmd_poly_d"],
        "mmd_max_reference": cfg["mmd_max_reference"],
        "mmd_norm": cfg["mmd_norm"],
    }


def _admit_fields(field2si, requested_fields):
    """Keep the requested fields whose interpolant is SDE (pos, cell) or DISCRETE (species).
    Raises if any requested field is not admissible (no silent downgrade)."""
    from omg.si.single_stochastic_interpolant import DifferentialEquationType
    admissible = {DifferentialEquationType.SDE, DifferentialEquationType.DISCRETE}
    admitted = tuple(f for f in requested_fields
                     if f in field2si and getattr(field2si[f], "_differential_equation_type", None) in admissible)
    dropped = tuple(f for f in requested_fields if f not in admitted)
    if dropped:
        available = [f for f, s in field2si.items()
                     if getattr(s, "_differential_equation_type", None) in admissible]
        raise RuntimeError(
            f"Requested fields {requested_fields}, but {dropped} have no stochastic (SDE) or discrete "
            f"interpolant in the model config. Learnable fields of this model: {available}")
    return admitted


def _provenance() -> Dict[str, Any]:
    info: Dict[str, Any] = {"argv": sys.argv}
    try:
        info["git_sha"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                         capture_output=True, text=True, check=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                                capture_output=True, text=True).stdout.strip())
    except Exception:
        info["git_sha"] = None
    from importlib import metadata
    versions = {}
    for pkg in ("torch", "lightning", "fairchem-core", "torch-sim-atomistic", "pymatgen", "ase", "numpy",
                "e3nn", "average-minimum-distance", "wandb"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info["versions"] = versions
    return info


def _lmdb_overlay(cfg, output_dir: Path) -> Path:
    """Model-config overlay that points the prior config's datasets at <data_dir>/mp_20."""
    import yaml
    mp20 = Path(cfg["data_dir"]) / "mp_20"
    overlay = {"data": {name: {"init_args": {"dataset": {"init_args": {"lmdb_paths": [str(mp20 / f"{split}.lmdb")]}}}}
                        for name, split in (("train_dataset", "train"), ("val_dataset", "val"),
                                            ("predict_dataset", "test"))}}
    path = output_dir / "data_overlay.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(overlay, f)
    return path


def main(argv: Optional[List[str]] = None) -> None:
    cfg = resolve_config(argv)
    if cfg["print_config"]:
        print(json.dumps({k: v for k, v in cfg.items() if k != "print_config"}, indent=2))
        return

    set_data_dir(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("init_checkpoint", "mmd_comp_reference", "creativity_reference"):
        if cfg[key] and not Path(cfg[key]).exists():
            raise FileNotFoundError(f"--{key} {cfg[key]} not found (the prior comes from scripts/download_assets.py, "
                                    f"the references from scripts/build_references.py)")

    # The seed is applied by LightningCLI when the model is built (seed_everything, workers=True).
    cli = OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
                 run=False, seed_everything_default=cfg["seed"],
                 args=[f"--config={cfg['model_config']}", f"--config={_lmdb_overlay(cfg, output_dir)}"])
    lm, dm = cli.model, cli.datamodule
    si, sampler, model = lm.si, lm.sampler, lm.model

    load_policy_weights(model, cfg["init_checkpoint"])

    dm.kwargs["batch_size"] = cfg["num_groups"]
    dm.kwargs["num_workers"] = 0
    dm.kwargs["persistent_workers"] = False
    if cfg["time_grid"] > si._integration_time_steps:
        warnings.warn(f"--time_grid {cfg['time_grid']} exceeds the model config's "
                      f"{si._integration_time_steps} points; using {si._integration_time_steps}")
    si._integration_time_steps = min(si._integration_time_steps, cfg["time_grid"])

    field2si = {df.name: s for df, s in zip(si._data_fields, si._stochastic_interpolants)}
    kwargs = module_kwargs(cfg)
    kwargs["fields"] = _admit_fields(field2si, kwargs["fields"])

    if "species" in field2si:
        field2si["species"]._noise = float(cfg["species_eta"])

    module = OMatGRPOModule(si=si, sampler=sampler, model=model, **kwargs)
    module.monitor_log = str(output_dir / "monitor.log")

    logger = WandbLogger(project=cfg["wandb_project"], name=cfg["run_name"], save_dir=str(output_dir),
                         log_model=False, mode=cfg["wandb_mode"])
    resolved = {"config": {k: v for k, v in cfg.items() if k != "print_config"}, **_provenance()}
    with open(output_dir / "resolved_config.json", "w") as f:
        json.dump(resolved, f, indent=2)
    logger.log_hyperparams(resolved["config"])

    steps_per_checkpoint = cfg["checkpoint_every"] * cfg["inner_epochs"]   # optimizer steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"), filename="step_{step:06d}",
        every_n_train_steps=steps_per_checkpoint, save_top_k=-1, save_last=True,
    )
    trainer = Trainer(
        max_steps=cfg["rollouts"] * cfg["inner_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=PRECISION,
        gradient_clip_val=None,   # the gradient norm is clipped manually in training_step
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback],
        logger=logger,
        enable_checkpointing=True,
        limit_train_batches=1.0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=dm.train_dataloader(), ckpt_path=cfg["resume"])

    final = output_dir / "final_model.ckpt"
    torch.save(module.model.state_dict(), final)
    print(f"Saved final model to {final}")
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in module.model.state_dict().items()},
                  str(output_dir / "final_model.safetensors"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
