"""Functional attribution for independent samples and [B, classes] scores."""
from dataclasses import dataclass
from itertools import product
import math

import torch
from torch.nn import functional as F

from .core import evaluating


def _inputs(inputs):
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise ValueError("inputs must be a floating tensor")
    if inputs.ndim < 2 or not inputs.numel() or not torch.isfinite(inputs).all():
        raise ValueError("inputs must be finite, nonempty and shaped [B, ...]")
    return inputs.detach().clone()


def _scores(output, target):
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError("Expected model output [batch, classes]")
    indices = torch.as_tensor(target, device=output.device)
    if indices.is_floating_point() or indices.dtype == torch.bool:
        raise ValueError("target must contain integer class indices")
    indices = indices.long()
    if indices.ndim == 0:
        indices = indices.expand(output.shape[0])
    if indices.shape != (output.shape[0],) or (indices < 0).any() or (indices >= output.shape[1]).any():
        raise ValueError("target must be a valid class index or one per sample")
    return output.gather(1, indices[:, None])[:, 0]


def _target(model, x, target):
    if target is not None:
        return target
    with torch.no_grad():
        output = model(x)
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("Expected model output [batch, classes]")
        return output.argmax(1)


def _baseline(x, baseline):
    try:
        value = torch.broadcast_to(torch.as_tensor(baseline, device=x.device, dtype=x.dtype), x.shape).clone()
    except (RuntimeError, TypeError) as exc:
        raise ValueError("baseline must broadcast to inputs") from exc
    if not torch.isfinite(value).all():
        raise ValueError("baseline must be finite")
    return value.detach()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class IntegratedGradientsResult:
    attributions: torch.Tensor
    completeness_delta: torch.Tensor


def integrated_gradients(model, inputs, target=None, *, baseline=0.0, steps=64):
    """Trapezoidal IG; delta = sum(attributions) - (score(x)-score(baseline))."""
    _positive_integer(steps, "steps")
    x = _inputs(inputs)
    base = _baseline(x, baseline)
    total = torch.zeros_like(x)
    with evaluating(model), torch.enable_grad():
        target = _target(model, x, target)
        for i in range(steps + 1):
            point = (base + (i / steps) * (x - base)).requires_grad_(True)
            score = _scores(model(point), target).sum()
            grad = torch.autograd.grad(score, point)[0]
            total += grad.detach() * (0.5 if i in (0, steps) else 1.0)
        attrs = (x - base) * total / steps
        with torch.no_grad():
            difference = _scores(model(x), target) - _scores(model(base), target)
        delta = attrs.flatten(1).sum(1) - difference
    return IntegratedGradientsResult(attrs.detach(), delta.detach())


def smoothgrad(model, inputs, target=None, *, samples=32, noise_std=0.1, generator=None):
    """Mean signed input gradients with absolute-unit Gaussian noise."""
    _positive_integer(samples, "samples")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    x = _inputs(inputs)
    total = torch.zeros_like(x)
    with evaluating(model), torch.enable_grad():
        target = _target(model, x, target)
        for _ in range(samples):
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            point = (x + noise_std * noise).requires_grad_(True)
            total += torch.autograd.grad(_scores(model(point), target).sum(), point)[0].detach()
    return total / samples


def occlusion(model, inputs, target=None, *, baseline=0.0, window_shape=None, strides=None):
    """Average score drop for covering windows over all nonbatch dimensions."""
    x = _inputs(inputs)
    base = _baseline(x, baseline)
    shape = x.shape[1:]
    window = tuple(window_shape) if window_shape is not None else (1,) * len(shape)
    stride = tuple(strides) if strides is not None else (1,) * len(shape)
    if len(window) != len(shape) or len(stride) != len(shape):
        raise ValueError("window_shape and strides must match nonbatch dimensions")
    for size, w, s in zip(shape, window, stride):
        _positive_integer(w, "window size")
        _positive_integer(s, "stride")
        if w > size or s > w:
            raise ValueError("Require stride <= window <= input dimension for full coverage")
    starts = []
    for size, w, s in zip(shape, window, stride):
        axis = list(range(0, size - w + 1, s))
        if axis[-1] != size - w:
            axis.append(size - w)
        starts.append(axis)
    total, count = torch.zeros_like(x), torch.zeros_like(x)
    with evaluating(model), torch.no_grad():
        target = _target(model, x, target)
        original = _scores(model(x), target)
        for position in product(*starts):
            region = (slice(None),) + tuple(slice(p, p + w) for p, w in zip(position, window))
            changed = x.clone()
            changed[region] = base[region]
            drop = original - _scores(model(changed), target)
            total[region] += drop.reshape((-1,) + (1,) * len(shape))
            count[region] += 1
    return total / count


def layercam(model, inputs, target=None, *, layer, normalize=True):
    """Positive local-gradient-weighted LayerCAM, resized to [B,H,W]."""
    x = _inputs(inputs)
    if x.ndim != 4:
        raise ValueError("LayerCAM requires [B,C,H,W] inputs")
    modules = dict(model.named_modules())
    if layer not in modules:
        raise ValueError(f"Unknown layer {layer!r}")
    captured = []

    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Selected layer must return a 4D tensor")
        value = output.clone()
        captured.append(value)
        return value.clone()

    handle = modules[layer].register_forward_hook(hook)
    try:
        with evaluating(model), torch.enable_grad():
            output = model(x.requires_grad_(True))
            if len(captured) != 1:
                raise ValueError("Selected layer must execute exactly once per forward")
            if target is None:
                if not isinstance(output, torch.Tensor) or output.ndim != 2:
                    raise ValueError("Expected model output [batch, classes]")
                target = output.detach().argmax(1)
            activation = captured[0]
            grad = torch.autograd.grad(_scores(output, target).sum(), activation)[0]
            cam = (activation * grad.relu()).sum(1).relu()
            cam = F.interpolate(cam[:, None], x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            if normalize:
                low = cam.amin((-2, -1), keepdim=True)
                span = cam.amax((-2, -1), keepdim=True) - low
                cam = (cam - low) / span.clamp_min(torch.finfo(cam.dtype).eps)
            return cam.detach()
    finally:
        handle.remove()
