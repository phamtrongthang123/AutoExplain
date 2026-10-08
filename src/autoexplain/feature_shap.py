"""Feature-level Shapley games, MM-SHAP aggregation and InterSHAP summaries.

Paper equations are documented in docs/feature-shap.md. The explicit replacement
and sampling choices here are not the papers' complete pretrained-model pipelines.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from math import comb

import torch

from .multimodal import MultimodalScorer, _inputs, _positive, _references


@dataclass(frozen=True)
class FeatureShapleyResult:
    feature_modalities: tuple[str, ...]
    contributions: torch.Tensor  # [B,G], signed
    baseline_scores: torch.Tensor  # [B]
    full_scores: torch.Tensor  # [B], all declared groups restored
    efficiency_residual: torch.Tensor  # [B]
    standard_error: torch.Tensor | None  # [B,G], permutation Monte Carlo only
    coalition_scores: torch.Tensor | None  # [B,2**G], exact method only
    method: str
    permutations: int


def feature_shapley(scorer, inputs, references, *, feature_masks,
                    feature_modalities, method="exact", permutations=128,
                    generator=None, max_features=64, max_coalitions=4096,
                    max_evaluations=1_000_000):
    """Joint feature game across modalities with exact or permutation estimates.

    feature_masks maps intervened input keys to integer masks broadcastable to
    their FULL input shape. Global IDs 0..G-1 identify players, -1 means fixed
    context. One player may span dependent fields (IDs and attention mask) using
    the same ID. feature_modalities[g] labels that player's semantic modality.
    References name exactly the intervened keys and obey MultimodalScorer's
    reference contract. Uncovered fields/elements remain original in every game.

    A common feature vocabulary is used across the batch; per-row absent groups
    are dummy players. Scores/targets must remain fixed and row-independent.
    Permutations are IID random feature orders, shared across batch rows, using
    the supplied generator's device. The returned standard error measures only
    permutation sampling, not model/reference/feature-selection uncertainty.
    This estimator differs from the paper's coalition-sampling configuration.
    """
    if not isinstance(scorer, MultimodalScorer):
        raise TypeError("scorer must be a MultimodalScorer")
    for value, name in ((max_features, "max_features"), (max_coalitions, "max_coalitions"),
                        (max_evaluations, "max_evaluations")):
        _positive(value, name)
    if method not in {"exact", "permutation"}:
        raise ValueError("method must be exact or permutation")
    if not isinstance(feature_modalities, (tuple, list)):
        raise ValueError("feature_modalities must be an explicit list/tuple of modality names")
    names = tuple(feature_modalities)
    count = len(names)
    if not 0 < count <= max_features or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("feature modality names must be nonempty and fit max_features")
    x = _inputs(inputs)
    if (not isinstance(feature_masks, Mapping) or not feature_masks
            or not set(feature_masks).issubset(x)):
        raise ValueError("feature_masks must name existing intervened input keys")
    bases = _references({key: x[key] for key in feature_masks}, references)
    masks, seen = {}, set()
    for key, value in feature_masks.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype not in (torch.int32, torch.int64)):
            raise ValueError("feature masks must be dense integer tensors")
        try:
            mask = torch.broadcast_to(value.detach().to(x[key].device), x[key].shape)
        except RuntimeError as exc:
            raise ValueError("feature masks must broadcast to full input shapes") from exc
        if (mask < -1).any() or (mask >= count).any():
            raise ValueError("feature IDs must lie in [-1,G-1]")
        seen.update(mask.unique().cpu().tolist())
        masks[key] = mask.clone()
    if seen - {-1} != set(range(count)):
        raise ValueError("every declared feature must occur in at least one mask")
    batch = next(iter(x.values())).shape[0]
    if method == "exact":
        coalitions = 1 << count
        if coalitions > max_coalitions or batch * coalitions > max_evaluations:
            raise ValueError("exact feature game exceeds coalition or scored-row budget")
    else:
        _positive(permutations, "permutations")
        if permutations < 2:
            raise ValueError("at least two permutations are needed for a standard error")
        if batch * (2 + permutations * (count - 1)) > max_evaluations:
            raise ValueError("permutation game exceeds scored-row budget")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    def score(active):
        changed = dict(x)
        for key, mask in masks.items():
            membership = torch.tensor(active, dtype=torch.bool, device=mask.device)
            keep = (mask == -1) | membership[mask.clamp_min(0)]
            changed[key] = torch.where(keep, x[key], bases[key])
        return scorer._score(changed)

    if method == "exact":
        scores = torch.stack([score([bool(bits & (1 << g)) for g in range(count)])
                              for bits in range(coalitions)], dim=1)
        base, full = scores[:, 0], scores[:, -1]
        contributions = scores.new_zeros((batch, count))
        for g in range(count):
            for bits in range(coalitions):
                if not bits & (1 << g):
                    contributions[:, g] += (scores[:, bits | (1 << g)] - scores[:, bits]) / (
                        count * comb(count - 1, bits.bit_count()))
        standard_error = None
        samples = 0
    else:
        scores = None
        base, full = score([False] * count), score([True] * count)
        contributions = base.new_zeros((batch, count))
        m2 = torch.zeros_like(contributions)
        device = generator.device if generator is not None else torch.device("cpu")
        for sample in range(permutations):
            order = torch.randperm(count, generator=generator, device=device).cpu().tolist()
            active, previous = [False] * count, base
            marginal = torch.zeros_like(contributions)
            for position, g in enumerate(order):
                active[g] = True
                current = full if position == count - 1 else score(active)
                marginal[:, g] = current - previous
                previous = current
            delta = marginal - contributions
            contributions += delta / (sample + 1)
            m2 += delta * (marginal - contributions)
        standard_error = (m2.clamp_min(0) / (permutations - 1) / permutations).sqrt()
        samples = permutations
    residual = contributions.sum(1) - (full - base)
    if (not torch.isfinite(contributions).all() or not torch.isfinite(residual).all()
            or (standard_error is not None and not torch.isfinite(standard_error).all())):
        raise ValueError("Shapley arithmetic overflowed; rescale the fixed score objective")
    return FeatureShapleyResult(names, contributions, base, full, residual,
                                standard_error, scores, method, samples)


def mm_shap(contributions, feature_modalities):
    """MM-SHAP equations (2)-(3): sum ABSOLUTE feature values, then normalize.

    This is NOT abs(sum(features)) or Shapley over whole modality coalitions.
    Input [B,G] is already-computed joint-game signed feature Shapley values.
    Returns per-sample modality masses/shares, valid flags and mean valid shares.
    Undefined all-zero rows have NaN shares and are explicitly excluded from the
    reported mean (valid_count is returned); this empty policy is our extension.
    """
    if (not isinstance(contributions, torch.Tensor) or contributions.ndim != 2
            or not contributions.is_floating_point() or min(contributions.shape) < 1
            or not torch.isfinite(contributions).all()):
        raise ValueError("contributions must be finite nonempty floating [B,G]")
    if not isinstance(feature_modalities, (list, tuple)):
        raise ValueError("feature_modalities must be a list/tuple")
    labels = tuple(feature_modalities)
    if (len(labels) != contributions.shape[1]
            or any(not isinstance(n, str) or not n for n in labels)):
        raise ValueError("one nonempty modality name is required per feature")
    names = tuple(dict.fromkeys(labels))
    values = contributions.detach().double().abs()
    scale = values.amax(1, keepdim=True)
    normalized = values / torch.where(scale > 0, scale, torch.ones_like(scale))
    mass = torch.stack([normalized[:, [i for i, label in enumerate(labels) if label == name]].sum(1)
                        for name in names], dim=1)
    total = mass.sum(1, keepdim=True)
    valid = total[:, 0] > 0
    shares = mass / torch.where(total > 0, total, torch.ones_like(total))
    shares = torch.where(valid[:, None], shares, torch.full_like(shares, float("nan")))
    absolute_mass = mass * scale
    if not torch.isfinite(absolute_mass).all():
        raise ValueError("absolute contribution mass overflowed; rescale the objective")
    mean = shares[valid].mean(0) if valid.any() else shares.new_full((len(names),), float("nan"))
    return {"modalities": names, "absolute_contributions": absolute_mass, "shares": shares,
            "valid": valid, "valid_count": int(valid.sum()), "mean_valid_shares": mean}


def intershap(interactions):
    """InterSHAP equations (5)-(11), given symmetric SHAP half-interactions.

    interactions [B,M,M] includes main effects on the diagonal. Off-diagonal
    entries each contain HALF the pair contribution (as in modality_shapley).
    Global = offdiag_sum(abs(mean_B(I))) / sum(abs(mean_B(I))).
    Local = offdiag_sum(abs(I_b)) / sum(abs(I_b)), then mean valid samples.
    Mean-before-absolute and absolute-before-mean are deliberately different.
    Supply the same scalar output semantics across samples; vector/class output
    reduction is not inferred. Zero-mass ratios are NaN with validity flags.
    """
    if (not isinstance(interactions, torch.Tensor) or interactions.ndim != 3
            or not interactions.is_floating_point() or min(interactions.shape) < 1
            or interactions.shape[1] != interactions.shape[2]
            or not torch.isfinite(interactions).all()):
        raise ValueError("interactions must be finite floating [B,M,M]")
    values = interactions.detach().double()
    if not torch.allclose(values, values.transpose(1, 2), rtol=1e-7, atol=1e-10):
        raise ValueError("interaction matrices must be symmetric under the half-pair convention")
    count = values.shape[1]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=values.device)

    def summarize(matrix):
        scale = matrix.amax(dim=(-2, -1), keepdim=True)
        normalized = matrix / torch.where(scale > 0, scale, torch.ones_like(scale))
        total = normalized.sum(dim=(-2, -1))
        valid = total > 0
        safe = torch.where(valid, total, torch.ones_like(total))
        ratio = normalized[..., off_diagonal].sum(-1) / safe
        modality = normalized.diagonal(dim1=-2, dim2=-1) / safe[..., None]
        ratio = torch.where(valid, ratio, torch.full_like(ratio, float("nan")))
        modality = torch.where(valid[..., None], modality, torch.full_like(modality, float("nan")))
        return ratio, modality, valid

    # Scaling avoids overflowing an intermediate sum while retaining cancellation.
    scale = values.abs().amax()
    mean = (values / torch.where(scale > 0, scale, torch.ones_like(scale))).mean(0) * scale
    global_matrix = mean.abs()
    global_score, global_modalities, global_valid = summarize(global_matrix)
    local, local_modalities, local_valid = summarize(values.abs())
    local_mean = local[local_valid].mean() if local_valid.any() else values.new_tensor(float("nan"))
    return {"global_intershap": global_score, "global_valid": global_valid,
            "absolute_mean_interactions": global_matrix,
            "global_modality_contributions": global_modalities,
            "local_intershap": local, "local_valid": local_valid,
            "local_modality_contributions": local_modalities,
            "mean_valid_local_intershap": local_mean, "local_valid_count": int(local_valid.sum())}
