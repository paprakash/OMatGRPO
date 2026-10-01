"""Parity of the recipes with the runs in the paper.

For each identifier, configs/recipes/<identifier>.yaml resolved by omg.grpo.train must give the
same module arguments, reward configuration and trainer settings as the original run. The
reference (tests/data/recipe_goldens/<identifier>.json) was captured from the unmodified training
entry point of the paper's runs (run_grpo_pilot.py) with the run's exact command line.

The rename map from the original flags to the current ones lives only here.
"""
import json
from pathlib import Path

import pytest

from omg.grpo.train import module_kwargs, resolve_config

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "data" / "recipe_goldens"
IDENTIFIERS = ["arityguard_creatrelax", "canonical_creatrelax", "sparseworst_creatrelax",
               "arityguard", "canonical", "sparseworst", "frozen_control"]

# original flag -> (current flag, value transform); None = removed (checked below)
_bool = lambda v: v.lower() in ("true", "1", "yes")
RENAME = {
    "--fields": ("fields", str),
    "--dng_mode": ("freeze_composition", lambda v: v.lower() == "true"),
    "--batch_size": ("num_groups", int),
    "--group_size": ("group_size", int),
    "--time_steps": ("time_grid", int),
    "--species_eta_override": ("species_eta", float),
    "--inner_epochs": ("inner_epochs", int),
    "--lr": ("lr", float),
    "--alpha_pos": ("alpha_pos", float),
    "--alpha_cell": ("alpha_cell", float),
    "--alpha_species": ("alpha_species", float),
    "--beta_kl": ("beta_kl_pos", float),
    "--beta_kl_species": ("beta_kl_species", float),
    "--w_ehull": ("w_stability", float),
    "--relax_before_reward": ("relax", _bool),
    "--relax_cell_dof": ("relax_cell", _bool),
    "--relax_max_steps": ("relax_steps", int),
    "--ehull_floor_at_zero": ("stability_floor_at_zero", _bool),
    "--ehull_cap": ("stability_cap", float),
    "--ehull_mag_floor": ("deep_below_hull", float),
    "--ehull_sparse_gate": ("sparse_gate", _bool),
    "--ehull_min_refset": ("sparse_min_refs", int),
    "--sparse_route": ("sparse_route", {"neutral": "abstain", "worst": "penalty"}.get),
    "--route_elemental": ("single_element_guard", {"worst": "on", "off": "off"}.get),
    "--use_diversity": ("occurrence_discount", _bool),
    "--div_tol": ("occurrence_tol", int),
    "--div_buff": ("occurrence_zero", int),
    "--mmd_kernel": ("mmd_kernel", str),
    "--mmd_norm": ("mmd_norm", str),
    "--w_creat": ("w_creat", float),
    "--creat_on_relaxed": ("creat_on_relaxed", lambda v: True if v is None else _bool(v)),
}
# original flags that no longer exist, with the value that makes them a no-op (the reward
# variants of the reward-hacking appendix are on the reward-hacking branch)
REMOVED_NOOP = {"--w_energy": "1.0", "--reward_type": "e_hull", "--w_rmsd_geom": "0.0", "--w_rmsd": "0.0"}
# flags that are only locations or names, not settings
NOT_SETTINGS = {"--config", "--ckpt_path", "--mmd_comp_reference", "--run_name", "--wandb_run_name"}


def _load(ident):
    golden = json.loads((GOLDENS / f"{ident}.json").read_text())
    cfg = resolve_config(["--recipe", str(REPO / "configs" / "recipes" / f"{ident}.yaml")],
                         env={"OMATGRPO_DATA_DIR": "/nonexistent/data"})
    return golden, cfg


@pytest.mark.parametrize("ident", IDENTIFIERS)
def test_module_kwargs_match_original_run(ident):
    golden, cfg = _load(ident)
    kw = module_kwargs(cfg)
    new_rc = kw.pop("reward_cfg")
    old = dict(golden["module_kwargs"])
    old_rc = dict(golden["reward_cfg"])

    # Removed arguments held no-op values in the original run.
    assert old.pop("ent_coef") == 0.0                      # entropy bonus, removed
    assert old_rc.pop("reward_mode") == "energy"           # sanity rewards, removed
    assert old_rc.pop("weights") == {"rmsd": 0.0, "energy": 1.0}   # --w_energy fixed at 1, no displacement reward
    # Reward-hacking variants, removed: the stability reward, and inert settings of the
    # residual geometry term and the displacement reward.
    assert old_rc.pop("reward_type") == "e_hull"
    assert old_rc.pop("w_rmsd_geom") == 0.0
    old_rc.pop("rmsd_geom_clamp"), old_rc.pop("fmax"), old.pop("reward_offset")

    # Locations: only whether they are set.
    assert (old.pop("mmd_comp_reference") is None) == (kw.pop("mmd_comp_reference") is None)
    old_rc.pop("creativity_reference")
    new_creat_ref = new_rc.pop("creativity_reference")
    assert (new_creat_ref is None) == (old_rc["w_creat"] == 0.0)

    # The MMD weight is inert when the bonus is off.
    if not old["use_mmd_diversity"]:
        assert not kw["use_mmd_diversity"]
        old.pop("w_mmd_diversity"), kw.pop("w_mmd_diversity")

    kw["fields"] = list(kw["fields"])
    assert kw == old
    assert new_rc == old_rc


@pytest.mark.parametrize("ident", IDENTIFIERS)
def test_trainer_settings_match_original_run(ident):
    from omg.grpo.train import GRAD_CLIP_NORM, PRECISION
    golden, cfg = _load(ident)
    assert cfg["rollouts"] * cfg["inner_epochs"] == golden["max_steps"]
    assert cfg["num_groups"] == golden["batch_size"]
    assert min(710, cfg["time_grid"]) == golden["integration_time_steps"]   # the prior config has 710
    assert cfg["species_eta"] == golden["species_noise"]
    assert PRECISION == golden["precision"]
    assert golden["gradient_clip_val"] is None and GRAD_CLIP_NORM == 0.5   # clipped manually
    assert cfg["seed"] == 0                                                  # LightningCLI default of the runs


@pytest.mark.parametrize("ident", IDENTIFIERS)
def test_every_original_flag_maps_to_the_same_value(ident):
    golden, cfg = _load(ident)
    for token in golden["argv_original"]:
        name, _, value = token.partition("=")
        value = value if "=" in token else None
        if name in NOT_SETTINGS or name == "--max_steps":
            continue
        if name == "--use_mmd_diversity":          # merged into --w_mmd (0 = off)
            assert (cfg["w_mmd"] > 0) == _bool(value), token
            continue
        if name == "--w_mmd_diversity":
            assert cfg["w_mmd"] == float(value), token
            continue
        if name in REMOVED_NOOP:
            assert value == REMOVED_NOOP[name], token
            continue
        assert name in RENAME, f"{ident}: original flag {name} has no mapping"
        new, transform = RENAME[name]
        assert cfg[new] == transform(value), f"{ident}: {token} -> {new}={cfg[new]!r}"
    max_steps = next(t for t in golden["argv_original"] if t.startswith("--max_steps="))
    assert cfg["rollouts"] * cfg["inner_epochs"] == int(max_steps.split("=")[1])


def test_defaults_equal_the_omatgrpo_recipe():
    base = resolve_config([], env={"OMATGRPO_DATA_DIR": "/nonexistent/data"})
    _, recipe = _load("arityguard_creatrelax")
    for key in ("run_name", "output_dir"):
        base.pop(key), recipe.pop(key)
    recipe.pop("recipe")
    base.pop("recipe")
    assert base == recipe


def test_unknown_flag_is_an_error():
    with pytest.raises(SystemExit):
        resolve_config(["--no_such_flag", "1"], env={})


def test_unknown_recipe_key_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("w_stabilty: 1.0\n")
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_config(["--recipe", str(bad)], env={})


def test_command_line_overrides_recipe():
    cfg = resolve_config(["--recipe", str(REPO / "configs/recipes/canonical.yaml"), "--w_mmd", "0"], env={})
    assert cfg["w_mmd"] == 0.0 and cfg["mmd_comp_reference"] is None
