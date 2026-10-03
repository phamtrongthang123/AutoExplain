"""Explicit task adapters; no automatic inference of segmentation objectives."""
import torch
from torch import nn


class SegmentationScoreAdapter(nn.Module):
    """Reduce [B,C,H,W] logits to [B,C] mean logits in a fixed spatial ROI.

    ``mask`` is a finite, nonnegative [H,W] weighting shared across examples.
    The adapter does not apply softmax or select the predicted segmentation.
    """

    def __init__(self, model: nn.Module, mask: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        if mask is not None:
            mask = torch.as_tensor(mask).detach().float().clone()
            if mask.ndim != 2 or not torch.isfinite(mask).all() or (mask < 0).any() or mask.sum() <= 0:
                raise ValueError("mask must be finite nonnegative [H,W] with positive mass")
        self.register_buffer("mask", mask)

    def forward(self, inputs):
        logits = self.model(inputs)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise ValueError("segmentation model must return [B,C,H,W] logits")
        if self.mask is None:
            return logits.mean((-2, -1))
        if self.mask.shape != logits.shape[-2:]:
            raise ValueError("mask spatial shape must match model logits")
        mask = self.mask.to(logits)
        return (logits * mask).sum((-2, -1)) / mask.sum()
