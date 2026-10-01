"""omg/grpo/checkpoints.py: every accepted checkpoint format loads the same weights, and a
checkpoint that does not fit the model raises instead of silently leaving random weights."""
import pytest
import torch

from omg.grpo.checkpoints import load_policy_weights, read_policy_state_dict


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 3)
        self.head = torch.nn.Linear(3, 2)


def _trained_state():
    torch.manual_seed(0)
    return _Net().state_dict()


def _assert_loaded(model, state):
    for k, v in state.items():
        assert torch.equal(model.state_dict()[k], v), k


def test_lightning_checkpoint(tmp_path):
    state = _trained_state()
    ckpt = {"state_dict": {**{f"model.{k}": v for k, v in state.items()},
                           **{f"model_ref.{k}": torch.zeros_like(v) for k, v in state.items()}},
            "epoch": 3, "global_step": 12}
    path = tmp_path / "lightning.ckpt"
    torch.save(ckpt, path)
    torch.manual_seed(1)
    model = _Net()
    load_policy_weights(model, path)
    _assert_loaded(model, state)   # model_ref.* is ignored, model.* is used


def test_bare_state_dict(tmp_path):
    """The final_model*.ckpt format: a bare state dict. It used to load nothing (bug a)."""
    state = _trained_state()
    path = tmp_path / "final_model.ckpt"
    torch.save(state, path)
    torch.manual_seed(1)
    model = _Net()
    load_policy_weights(model, path)
    _assert_loaded(model, state)


def test_safetensors(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    state = _trained_state()
    path = tmp_path / "final_model.safetensors"
    safetensors_torch.save_file(state, str(path))
    torch.manual_seed(1)
    model = _Net()
    load_policy_weights(model, path)
    _assert_loaded(model, state)
    assert read_policy_state_dict(path).keys() == state.keys()


def test_no_matching_key_raises(tmp_path):
    path = tmp_path / "other.ckpt"
    torch.save({"something.weight": torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="no parameter name matches"):
        load_policy_weights(_Net(), path)


def test_missing_key_raises(tmp_path):
    state = _trained_state()
    state.pop("head.bias")
    path = tmp_path / "partial.ckpt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="1 missing"):
        load_policy_weights(_Net(), path)
