"""Scoped tensor activation capture and replacement (not concurrent-safe)."""
from contextlib import contextmanager

import torch


def _module(model, layer):
    modules = dict(model.named_modules())
    if layer not in modules:
        raise ValueError(f"Unknown layer {layer!r}")
    return modules[layer]


@contextmanager
def capture_activations(model, layers):
    """Yield {layer: [detached tensor per invocation]}; preserve mode and gradients."""
    names = [layers] if isinstance(layers, str) else list(layers)
    if not names or len(set(names)) != len(names):
        raise ValueError("layers must be nonempty and unique")
    modules = {name: _module(model, name) for name in names}
    values = {name: [] for name in names}
    handles = []

    def capture(name):
        def hook(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise ValueError("Capture requires tensor outputs")
            values[name].append(output.detach().clone())
        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        yield values
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_activation(model, layer, replacement, *, mask=None):
    """Replace tensor output, optionally at a broadcast boolean mask; no in-place edits.

    replacement must exactly match each output's shape/device/dtype. It is cloned
    and detached on entry. Contexts do not change model train/eval state.
    """
    module = _module(model, layer)
    if not isinstance(replacement, torch.Tensor):
        raise ValueError("replacement must be a tensor")
    value = replacement.detach().clone()
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor")
        mask = mask.detach().clone()

    def patch(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("Patching requires tensor outputs")
        if output.shape != value.shape or output.device != value.device or output.dtype != value.dtype:
            raise ValueError("replacement must match output shape, device and dtype")
        if mask is None:
            return value.clone()
        try:
            selected = torch.broadcast_to(mask.to(output.device), output.shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to output shape") from exc
        return torch.where(selected, value, output)

    handle = module.register_forward_hook(patch)
    try:
        yield model
    finally:
        handle.remove()
