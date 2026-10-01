"""omg.grpo.train: configuration validation, flag parsing, --print_config, the --resume wiring, the
KL-reference check on resume, and training with zero KL weights."""
import json
import math
import types

import pytest
import torch
import lightning as L

from omg.grpo import train
from omg.grpo.grpo_lightning import OMatGRPOModule

ENV = {"OMATGRPO_DATA_DIR": "/nonexistent/data"}


def _cfg(*argv):
    return train.resolve_config(list(argv), env=ENV)


# ------------------------------- validation -------------------------------

@pytest.mark.parametrize("argv, message", [
    (["--freeze_composition", "true", "--fields", "pos,cell,species", "--occurrence_discount", "false"],
     "needs --fields pos,cell"),
    (["--freeze_composition", "true", "--fields", "pos,cell"], "needs --occurrence_discount false"),
    (["--relax", "false"], "--creat_on_relaxed true needs --relax true"),
    (["--stability_cap", "0"], "--stability_cap must be > 0"),
    (["--group_size", "1"], "--group_size must be >= 2"),
    (["--occurrence_tol", "6"], "--occurrence_tol must be smaller"),
    (["--species_eta", "0.2"], "needs --species_eta 0"),
    (["--w_mmd", "-1"], "--w_mmd must be >= 0"),
])
def test_invalid_combinations_are_rejected(argv, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _cfg(*argv)


def test_fields_must_be_an_allowed_combination():
    with pytest.raises(SystemExit):
        _cfg("--fields", "species")


def test_valid_frozen_control_and_relax_off():
    cfg = _cfg("--freeze_composition", "true", "--fields", "pos,cell", "--occurrence_discount", "false",
               "--beta_kl_species", "0")
    assert train.module_kwargs(cfg)["dng_mode"] is True
    cfg = _cfg("--relax", "false", "--creat_on_relaxed", "false")
    assert train.module_kwargs(cfg)["reward_cfg"]["relax_before_reward"] is False


def test_beta_kl_species_without_species_warns():
    with pytest.warns(UserWarning, match="no effect"):
        _cfg("--fields", "pos,cell")


# ------------------------------- parsing -------------------------------

def test_boolean_forms():
    assert _cfg("--relax_cell", "false")["relax_cell"] is False
    assert _cfg("--relax_cell", "0")["relax_cell"] is False
    assert _cfg("--relax_cell")["relax_cell"] is True        # bare flag
    with pytest.raises(SystemExit):
        _cfg("--relax_cell", "maybe")


def test_deep_below_hull_accepts_minus_inf():
    cfg = _cfg("--deep_below_hull=-inf")          # "=" form: argparse reads a bare "-inf" as an option
    assert cfg["deep_below_hull"] == -math.inf
    assert train.module_kwargs(cfg)["reward_cfg"]["ehull_mag_floor"] == -math.inf


def test_mmd_off_passes_no_reference():
    kw = train.module_kwargs(_cfg("--w_mmd", "0"))
    assert kw["use_mmd_diversity"] is False and kw["mmd_comp_reference"] is None


def test_print_config(capsys):
    train.main(["--print_config", "--w_creat", "0.5"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["w_creat"] == 0.5 and printed["rollouts"] == 750


# ------------------------------- resume -------------------------------

class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)


def _bare_module(ref_state):
    m = OMatGRPOModule.__new__(OMatGRPOModule)
    L.LightningModule.__init__(m)
    m.model_ref = _Net()
    m.model_ref.load_state_dict(ref_state)
    return m


def test_resume_keeps_the_prior_as_kl_reference():
    torch.manual_seed(0)
    prior = _Net().state_dict()
    m = _bare_module(prior)
    same = {"state_dict": {f"model_ref.{k}": v.clone() for k, v in prior.items()}}
    m.on_load_checkpoint(same)                       # same reference: accepted
    other = {"state_dict": {f"model_ref.{k}": v + 1 for k, v in prior.items()}}
    with pytest.raises(ValueError, match="different KL reference"):
        m.on_load_checkpoint(other)


def test_resume_is_passed_to_trainer_fit(tmp_path, monkeypatch):
    """--resume reaches Trainer.fit(ckpt_path=...), which restores policy, optimizer and step."""
    calls = {}

    class _SI:
        _integration_time_steps = 710
        _data_fields = []
        _stochastic_interpolants = []

    class _DM:
        kwargs = {}

        def train_dataloader(self):
            return "loader"

    class _CLI:
        def __init__(self, *a, **kw):
            calls["seed"] = kw["seed_everything_default"]
            self.model = types.SimpleNamespace(si=_SI(), sampler=None, model=_Net())
            self.datamodule = _DM()

    class _Module:
        def __init__(self, **kw):
            self.model = kw["model"]

    class _Logger:
        def __init__(self, *a, **kw):
            pass

        def log_hyperparams(self, *a):
            pass

    class _Trainer:
        def __init__(self, *a, **kw):
            calls["max_steps"] = kw["max_steps"]

        def fit(self, module, train_dataloaders=None, ckpt_path=None):
            calls["ckpt_path"] = ckpt_path

    monkeypatch.setattr(train, "OMGCLI", _CLI)
    monkeypatch.setattr(train, "load_policy_weights", lambda model, path: None)
    monkeypatch.setattr(train, "_admit_fields", lambda f2s, fields: fields)
    monkeypatch.setattr(train, "OMatGRPOModule", _Module)
    monkeypatch.setattr(train, "WandbLogger", _Logger)
    monkeypatch.setattr(train, "Trainer", _Trainer)
    for name in ("prior.safetensors", "mmd.pt", "creat.json.gz", "last.ckpt"):
        (tmp_path / name).write_bytes(b"x")
    train.main(["--output_dir", str(tmp_path / "out"), "--init_checkpoint", str(tmp_path / "prior.safetensors"),
                "--mmd_comp_reference", str(tmp_path / "mmd.pt"),
                "--creativity_reference", str(tmp_path / "creat.json.gz"),
                "--resume", str(tmp_path / "last.ckpt"), "--rollouts", "2"])
    assert calls == {"seed": 0, "max_steps": 6, "ckpt_path": str(tmp_path / "last.ckpt")}
    resolved = json.loads((tmp_path / "out" / "resolved_config.json").read_text())
    assert resolved["config"]["resume"] == str(tmp_path / "last.ckpt") and "git_sha" in resolved


# ------------------------------- zero KL weights -------------------------------

def test_zero_kl_weights_still_backpropagate():
    """--beta_kl_pos 0 and --beta_kl_species 0 must train: the loss and the diagnostic KL total
    stay differentiable."""
    m = OMatGRPOModule.__new__(OMatGRPOModule)
    L.LightningModule.__init__(m)
    m.fields = ("pos",)
    m.eps_clip, m.alpha_pos, m.alpha_cell, m.alpha_species = 0.2, 1.0, 1.0, 1.0
    m.beta_kl_pos = m.beta_kl = 0.0
    m.beta_kl_species = 0.0
    new = torch.zeros(4, 3, requires_grad=True)
    kl_pos = (new ** 2).sum()                       # stands in for a KL that depends on the policy
    loss, stats = m._ppo_grpo_loss({"pos": torch.zeros(4, 3)}, {"pos": new}, torch.tensor([1., -1., .5, 0.]),
                                   analytical_kl={"pos": kl_pos, "species": torch.tensor(0.0)})
    loss.backward()
    assert new.grad is not None
    kl_total_t = m.beta_kl_pos * kl_pos + m.beta_kl_species * torch.tensor(0.0)
    assert kl_total_t.requires_grad


def test_flag_table_lists_every_flag():
    table = train.flag_table_markdown()
    for action in train.build_parser()._actions:
        if action.dest != "help":
            assert f"`--{action.dest}`" in table, action.dest
