"""SAE feature inspection and error-preserving interventions.

Supports instantiated OpenAI SAEs and SAELens SAEs. No weight downloads occur.
"""
from contextlib import contextmanager
import math
import torch


class SAEAdapter:
    def __init__(self, sae, *, backend="saelens"):
        if backend not in {"saelens", "openai"}:
            raise ValueError("backend must be saelens or openai")
        self.sae, self.backend = sae, backend

    def encode(self, x):
        result = self.sae.encode(x)
        return result if self.backend == "openai" else (result, None)

    def decode(self, features, context=None):
        return self.sae.decode(features, context) if self.backend == "openai" else self.sae.decode(features)

    @torch.no_grad()
    def inspect(self, x, *, top_k=5):
        z, context = self.encode(x)
        reconstruction = self.decode(z, context)
        if not z.isfinite().all() or not reconstruction.isfinite().all():
            raise ValueError("Nonfinite SAE output")
        if not 1 <= top_k <= z.shape[-1]:
            raise ValueError("top_k outside feature count")
        values, indices = z.topk(top_k, dim=-1)
        return {"features": z, "reconstruction": reconstruction,
                "reconstruction_mse": (x - reconstruction).square().mean().item(),
                "mean_l0": (z != 0).float().sum(-1).mean().item(),
                "top_values": values, "top_indices": indices}

    def edit(self, x, *, feature, value=None, scale=None, preserve_error=True):
        if (value is None) == (scale is None):
            raise ValueError("Provide exactly one of value or scale")
        magnitude = value if value is not None else scale
        if not math.isfinite(magnitude):
            raise ValueError("Intervention magnitude must be finite")
        z, context = self.encode(x)
        if type(feature) is not int or not 0 <= feature < z.shape[-1]:
            raise ValueError("Feature index outside SAE")
        changed = z.clone()
        changed[..., feature] = value if value is not None else z[..., feature] * scale
        decoded = self.decode(changed, context)
        # Zero intervention is identity even when reconstruction is imperfect.
        return x + decoded - self.decode(z, context) if preserve_error else decoded


@contextmanager
def steer_sae(model, layer, adapter, *, feature, value=None, scale=None, preserve_error=True):
    module = dict(model.named_modules()).get(layer)
    if module is None:
        raise ValueError("Unknown layer")
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("SAE hook requires tensor output; select an explicit adapter for tuples")
        return adapter.edit(output, feature=feature, value=value, scale=scale, preserve_error=preserve_error)
    handle = module.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def load_saelens_local(directory, *, device="cpu"):
    """Load caller-provided SAELens-format weights/config without remote access."""
    from pathlib import Path
    from .backends import require
    if not Path(directory).is_dir():
        raise ValueError("A local SAE directory is required")
    backend = require("sae_lens", "sae")
    return SAEAdapter(backend.SAE.load_from_disk(directory, device=device), backend="saelens")


def load_saelens_release(release, sae_id, *, allow_download=False, device="cpu"):
    """Explicit opt-in to upstream pretrained loading, including Gemma Scope releases.

    User must accept applicable artifact/base-model terms separately. This does
    not accept licenses, log into services or download a base model for the user.
    Release and exact sae_id are caller-selected; no guessed default artifact.
    """
    if not allow_download:
        raise PermissionError("Pretrained SAE retrieval requires allow_download=True")
    from .backends import require
    backend = require("sae_lens", "sae")
    return SAEAdapter(backend.SAE.from_pretrained(release, sae_id, device=device), backend="saelens")


def feature_examples(features, feature, *, k=5):
    """Top observation/token positions for one feature; no automatic semantic label."""
    if features.ndim < 2 or not 0 <= feature < features.shape[-1] or k < 1:
        raise ValueError("Invalid feature data/index/k")
    values = features[..., feature].detach().flatten()
    scores, indices = values.topk(min(k, values.numel()))
    return {"flat_indices": indices, "scores": scores, "position_shape": features.shape[:-1]}
