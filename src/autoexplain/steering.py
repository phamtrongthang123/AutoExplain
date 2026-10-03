from contextlib import contextmanager
import math

import torch
from torch import nn


def concept_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Unit mean-difference direction from [observations, features] activations.

    Callers are responsible for matched conditions, layer/timestep selection, and
    train/evaluation separation. Flattening spatial tokens is an explicit caller choice.
    """
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
        raise ValueError("Expected [observations, features] tensors with equal feature width")
    if not positive.shape[0] or not negative.shape[0]:
        raise ValueError("Activation sets must be nonempty")
    direction = positive.detach().float().mean(0) - negative.detach().float().mean(0)
    norm = direction.norm()
    if not torch.isfinite(direction).all() or norm <= 1e-12:
        raise ValueError("Concept direction is zero or nonfinite")
    return direction / norm


@contextmanager
def steer(model: nn.Module, layer: str, direction: torch.Tensor, *, strength: float = 1.0,
          feature_axis: int = -1, mode: str = "add"):
    """Temporarily intervene on a tensor-output module.

    add: h' = h + strength*v
    remove_positive_projection: h' = h - strength*max(<h,v>,0)*v

    This is a generic primitive, NOT a full PolypSteer reproduction. The caller
    must select conditional branches, attention sites, timestep windows and masks.
    Hooks always detach on context exit, including when forward raises.
    Do not share a hooked model across concurrent requests.
    """
    layers = dict(model.named_modules())
    if layer not in layers:
        raise ValueError(f"Unknown layer {layer!r}")
    if mode not in {"add", "remove_positive_projection"}:
        raise ValueError("Unknown steering mode")
    if direction.ndim != 1 or not direction.numel() or not torch.isfinite(direction).all():
        raise ValueError("direction must be a finite, nonempty vector")
    vector = direction.detach().float().clone()
    norm = vector.norm()
    if norm <= 1e-12 or not math.isfinite(strength):
        raise ValueError("Require a nonzero direction and finite strength")
    vector /= norm

    def intervene(module, args, output):
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise ValueError("Steering requires a floating tensor output; wrap structured outputs")
        if not -output.ndim <= feature_axis < output.ndim:
            raise ValueError("feature_axis is outside output dimensions")
        axis = feature_axis % output.ndim
        if output.shape[axis] != vector.numel():
            raise ValueError("Direction width does not match the selected feature axis")
        shape = [1] * output.ndim
        shape[axis] = vector.numel()
        v = vector.to(output).reshape(shape)
        if mode == "add":
            return output + strength * v
        projection = (output * v).sum(dim=axis, keepdim=True).clamp_min(0)
        return output - strength * projection * v

    handle = layers[layer].register_forward_hook(intervene)
    try:
        yield model
    finally:
        handle.remove()
