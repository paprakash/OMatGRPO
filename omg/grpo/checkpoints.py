"""Loading policy weights from the checkpoint formats OMatGRPO uses."""
from pathlib import Path
from typing import Dict

import torch


def read_policy_state_dict(path) -> Dict[str, torch.Tensor]:
    """Return the state dict of the policy network (keys as in ``Model.state_dict()``).

    Accepted formats:
      - a Lightning checkpoint, whose ``state_dict`` holds the policy under the ``model.`` prefix
        (the pretrained prior, and periodic training checkpoints, which also hold ``model_ref.*``);
      - a bare ``state_dict`` of the policy (the ``final_model*.ckpt`` files written at the end of
        training);
      - a ``.safetensors`` file holding a bare state dict (the released weights).
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Lightning checkpoints can carry objects the safe unpickler rejects.
            obj = torch.load(path, map_location="cpu", weights_only=False)
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(state).__name__}")
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    return state


def load_policy_weights(model: torch.nn.Module, path) -> Dict[str, int]:
    """Load policy weights into ``model`` and fail loudly unless every key matches.

    A checkpoint that matches no key, or leaves keys missing or unexpected, raises ValueError,
    so a wrong file can never silently leave the model at its random initialization.
    """
    state = read_policy_state_dict(path)
    expected = set(model.state_dict().keys())
    got = set(state.keys())
    if not expected & got:
        raise ValueError(
            f"{path}: no parameter name matches the model (checkpoint keys start with "
            f"{sorted(got)[:3]}, model keys with {sorted(expected)[:3]})."
        )
    missing, unexpected = sorted(expected - got), sorted(got - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path}: {len(missing)} missing and {len(unexpected)} unexpected keys "
            f"(missing {missing[:5]}, unexpected {unexpected[:5]})."
        )
    model.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} tensors into the policy from {path}")
    return {"n_tensors": len(state)}
