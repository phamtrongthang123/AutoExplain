"""Matched-noise diffusion sampling with explicit per-step activation policies."""
from contextlib import contextmanager
import math
import torch


@contextmanager
def diffusion_intervention(denoiser, layer, direction, *, strength=1.0,
                           mode="add", feature_axis=1, batch_indices=None, spatial_mask=None):
    """Hook a tensor site during a caller-selected timestep.

    Explicit batch_indices select conditional rows; None means ALL rows. There
    is no assumption that classifier-free guidance uses any particular ordering.
    spatial_mask must broadcast to the activation with singleton feature axis.
    This is a primitive, not an exact SSPS/PolypSteer reproduction.
    """
    module = dict(denoiser.named_modules()).get(layer)
    if module is None or mode not in {"add", "remove_positive_projection"}:
        raise ValueError("Invalid layer/mode")
    v = direction.detach().float()
    if v.ndim != 1 or not v.isfinite().all() or v.norm() <= 1e-12 or not math.isfinite(strength):
        raise ValueError("Invalid direction/strength")
    v = v / v.norm()
    if batch_indices is not None and (not batch_indices or any(type(i) is not int or i < 0 for i in batch_indices)):
        raise ValueError("Need nonnegative batch indices")
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise ValueError("Choose a floating tensor-output module")
        if not -output.ndim <= feature_axis < output.ndim:
            raise ValueError("Invalid feature_axis")
        axis = feature_axis % output.ndim
        if axis == 0 or output.shape[axis] != v.numel():
            raise ValueError("Feature width mismatch or batch feature axis")
        shape = [1] * output.ndim
        shape[axis] = v.numel()
        vector = v.to(output).reshape(shape)
        delta = strength * vector.expand_as(output)
        if mode == "remove_positive_projection":
            delta = -delta * (output * vector).sum(axis, keepdim=True).clamp_min(0)
        mask = torch.ones_like(output)
        if batch_indices is not None:
            if max(batch_indices) >= output.shape[0]:
                raise ValueError("Batch index outside activation")
            mask.zero_()
            mask[batch_indices] = 1
        if spatial_mask is not None:
            m = torch.as_tensor(spatial_mask, device=output.device, dtype=output.dtype)
            if not m.isfinite().all() or (m < 0).any() or (m > 1).any():
                raise ValueError("Mask must be finite in [0,1]")
            expanded = torch.broadcast_to(m, output.shape)
            if not torch.equal(expanded, expanded.select(axis, 0).unsqueeze(axis).expand_as(output)):
                raise ValueError("Spatial mask must be constant across feature channels")
            mask = mask * expanded
        return output + delta * mask
    handle = module.register_forward_hook(hook)
    try:
        yield denoiser
    finally:
        handle.remove()


def sample_diffusers(denoiser, scheduler, initial_noise, *, steps=8, seed=0,
                     intervention=None, active_steps=(), model_kwargs=None):
    """Sample a diffusers epsilon/sample predictor using caller-provided scheduler.

    intervention is a zero-argument context factory; active_steps are loop indices,
    not raw scheduler timesteps. Fresh generator + same initial_noise yield matched
    controls. Caller supplies independent scheduler instances for comparisons.
    Does not perform text encoding, VAE decoding or classifier-free guidance.
    """
    from contextlib import nullcontext
    from .core import evaluating
    if not 1 <= steps <= 100 or any(type(i) is not int or i < 0 or i >= steps for i in active_steps):
        raise ValueError("Invalid sampling budget or active steps")
    generator = torch.Generator(device=initial_noise.device).manual_seed(seed)
    scheduler.set_timesteps(steps, device=initial_noise.device)
    sample = initial_noise.detach().clone() * scheduler.init_noise_sigma
    with evaluating(denoiser), torch.no_grad():
        for index, timestep in enumerate(scheduler.timesteps):
            context = intervention() if intervention is not None and index in active_steps else nullcontext()
            with context:
                scaled = scheduler.scale_model_input(sample, timestep)
                output = denoiser(scaled, timestep, **(model_kwargs or {}))
                prediction = output.sample if hasattr(output, "sample") else output
            sample = scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
            if not sample.isfinite().all():
                raise ValueError("Nonfinite diffusion sample")
    return sample.detach()
