"""Scalar-feature deletion/insertion curves with matched random order controls."""
from dataclasses import dataclass

import torch

from .attribution import _baseline, _inputs, _positive_integer, _scores, _target
from .core import evaluating


@dataclass(frozen=True)
class FaithfulnessResult:
    fractions: torch.Tensor
    scores: torch.Tensor
    auc: torch.Tensor
    random_scores: torch.Tensor
    random_auc: torch.Tensor


def perturbation_faithfulness(model, inputs, attributions, target=None, *, mode="deletion",
                             baseline=0.0, steps=20, random_trials=5, generator=None):
    """Rank absolute scalar attributions; return raw-score curves and random controls.

    scores [B,K], auc [B], random_scores [R,B,K], random_auc [R,B].
    K = min(steps, number of scalar features) + 1, including both endpoints.
    """
    _positive_integer(steps, "steps")
    _positive_integer(random_trials, "random_trials")
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be deletion or insertion")
    x = _inputs(inputs)
    if not isinstance(attributions, torch.Tensor) or attributions.shape != x.shape:
        raise ValueError("attributions must have inputs shape")
    if not torch.isfinite(attributions).all():
        raise ValueError("attributions must be finite")
    base = _baseline(x, baseline)
    features = x[0].numel()
    counts = torch.linspace(0, features, min(steps, features) + 1, device=x.device).round().long()
    fractions = counts.to(x.dtype) / features
    order = attributions.detach().to(x.device).abs().flatten(1).argsort(dim=1, descending=True, stable=True)
    flat, flat_base = x.flatten(1), base.flatten(1)

    def curve(ranking, selected_target):
        values = []
        for count in counts.tolist():
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(1, ranking[:, :count], True)
            if mode == "deletion":
                changed = torch.where(mask, flat_base, flat)
            else:
                changed = torch.where(mask, flat, flat_base)
            values.append(_scores(model(changed.reshape_as(x)), selected_target))
        return torch.stack(values, dim=1)

    with evaluating(model), torch.no_grad():
        target = _target(model, x, target)
        scores = curve(order, target)
        controls = []
        for _ in range(random_trials):
            ranking = torch.stack([torch.randperm(features, device=x.device, generator=generator)
                                   for _ in range(x.shape[0])])
            controls.append(curve(ranking, target))
        random_scores = torch.stack(controls)
        auc = torch.trapezoid(scores, fractions, dim=-1)
        random_auc = torch.trapezoid(random_scores, fractions, dim=-1)
    return FaithfulnessResult(fractions, scores, auc, random_scores, random_auc)
