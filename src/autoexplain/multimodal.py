"""Forward-only named-modality games and empirical additive projections.

These are explicit mathematical primitives, not end-to-end paper reproductions.
See ``docs/multimodal-contributions.md`` for score and replacement assumptions.
"""
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from math import comb

import torch
from torch import nn

from .core import evaluating


@dataclass(frozen=True)
class ModalityShapleyResult:
    names: tuple[str, ...]
    coalition_scores: torch.Tensor  # [B, 2**M], bit positions follow names
    contributions: torch.Tensor  # [B, M]
    interactions: torch.Tensor  # [B, M, M], symmetric half-interaction convention
    efficiency_residual: torch.Tensor  # [B]


@dataclass(frozen=True)
class ModalityAblationResult:
    names: tuple[str, ...]
    full_scores: torch.Tensor  # [B]
    reference_scores: torch.Tensor  # [B]
    without_scores: torch.Tensor  # [B, M]
    only_scores: torch.Tensor  # [B, M]
    removal_effects: torch.Tensor  # full - without
    addition_effects: torch.Tensor  # only - reference


@dataclass(frozen=True)
class MismatchedPairResult:
    modality: str
    permutation: torch.Tensor  # [B], changed row i takes original row permutation[i]
    matched_scores: torch.Tensor  # [B]
    mismatched_scores: torch.Tensor  # [B]
    score_drop: torch.Tensor  # matched - mismatched


@dataclass(frozen=True)
class EMAPResult:
    names: tuple[str, str]
    pair_scores: torch.Tensor  # [N, N], first modality row i, second row j
    grand_mean: torch.Tensor  # scalar
    first_effects: torch.Tensor  # [N], row mean - grand mean
    second_effects: torch.Tensor  # [N], column mean - grand mean
    additive_scores: torch.Tensor  # [N, N]
    residuals: torch.Tensor  # [N, N]
    matched_scores: torch.Tensor  # [N], original paired observations
    matched_additive_scores: torch.Tensor  # [N]


@dataclass(frozen=True)
class RelianceComparison:
    answer: ModalityShapleyResult
    explanation: ModalityShapleyResult
    answer_shares: torch.Tensor  # [B, M], normalized absolute contributions
    explanation_shares: torch.Tensor  # [B, M]
    valid: torch.Tensor  # [B], both absolute totals nonzero
    l1_distance: torch.Tensor  # [B], NaN for undefined zero-total rows


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _inputs(inputs):
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("inputs must be a nonempty mapping of modality names to tensors")
    result = {}
    batch = device = None
    for name, value in inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("modality names must be nonempty strings")
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.ndim < 1 or not value.numel() or value.is_complex()):
            raise ValueError(f"{name} must be a nonempty real dense batched tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        if batch is None:
            batch, device = value.shape[0], value.device
        if value.shape[0] != batch or value.device != device:
            raise ValueError("all modalities must share batch size and device")
        result[name] = value.detach().clone()
    return result


def _references(inputs, references):
    if not isinstance(references, Mapping) or set(references) != set(inputs):
        raise ValueError("references must specify exactly the input modality names")
    result = {}
    for name, value in inputs.items():
        reference = references[name]
        if (not isinstance(reference, torch.Tensor) or reference.layout != torch.strided
                or reference.device != value.device or reference.dtype != value.dtype):
            raise ValueError(f"reference {name} must have input dtype, device and dense layout")
        if reference.shape not in (value.shape, (1, *value.shape[1:])):
            raise ValueError(f"reference {name} must have input shape or a singleton batch")
        if not torch.isfinite(reference).all():
            raise ValueError(f"reference {name} must be finite")
        result[name] = reference.detach().expand_as(value).clone()
    return result


class MultimodalScorer:
    """Adapt ``score_fn(dict[str, Tensor]) -> finite floating Tensor[B]``.

    Provide ``model`` for a closure around an nn.Module so evaluation mode is
    managed. An nn.Module score_fn is managed automatically. Scores are detached
    float64 tensors on the input device; no target selection is inferred.
    """

    def __init__(self, score_fn: Callable, *, model: nn.Module | None = None):
        if not callable(score_fn):
            raise TypeError("score_fn must be callable")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module or None")
        self.score_fn = score_fn
        self.model = model

    def _score(self, inputs):
        first = next(iter(inputs.values()))
        # Isolate inputs from accidental in-place writes by caller adapters.
        with ExitStack() as stack:
            if isinstance(self.score_fn, nn.Module):
                stack.enter_context(evaluating(self.score_fn))
            if self.model is not None and self.model is not self.score_fn:
                stack.enter_context(evaluating(self.model))
            stack.enter_context(torch.no_grad())
            output = self.score_fn({name: value.clone() for name, value in inputs.items()})
            if (not isinstance(output, torch.Tensor) or not output.is_floating_point()
                    or output.layout != torch.strided or output.shape != (first.shape[0],)
                    or output.device != first.device):
                raise ValueError("score_fn must return floating [batch] scores on the input device")
            if not torch.isfinite(output).all():
                raise ValueError("score_fn returned nonfinite scores")
            return output.detach().to(dtype=torch.float64).clone()

    def __call__(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._score(_inputs(inputs))


def _scorer(scorer):
    if not isinstance(scorer, MultimodalScorer):
        raise TypeError("scorer must be a MultimodalScorer with an explicit scalar objective")


def _game(scorer, inputs, references, masks, max_evaluations):
    _positive(max_evaluations, "max_evaluations")
    masks = tuple(dict.fromkeys(masks))
    names = tuple(inputs)
    batch = next(iter(inputs.values())).shape[0]
    if len(masks) * batch > max_evaluations:
        raise ValueError("requested coalition rows exceed max_evaluations")
    return {mask: scorer._score({name: inputs[name] if mask & (1 << i) else references[name]
                                for i, name in enumerate(names)}) for mask in masks}


def modality_shapley(scorer, inputs, references, *, max_coalitions=256,
                     max_evaluations=1_000_000) -> ModalityShapleyResult:
    """Exact fixed-reference Shapley values and SHAP pair interaction matrix.

    Whole named modalities are players. The default permits at most 8 players.
    Off-diagonals are half the weighted second difference; each row sums to its
    Shapley value after setting the diagonal to its remaining main effect.
    """
    _scorer(scorer)
    _positive(max_coalitions, "max_coalitions")
    inputs = _inputs(inputs)
    references = _references(inputs, references)
    names, m = tuple(inputs), len(inputs)
    count = 1 << m
    if count > max_coalitions:
        raise ValueError("exact modality game exceeds max_coalitions")
    values = _game(scorer, inputs, references, range(count), max_evaluations)
    scores = torch.stack([values[mask] for mask in range(count)], dim=1)
    phi = scores.new_zeros((scores.shape[0], m))
    interaction = scores.new_zeros((scores.shape[0], m, m))
    for i in range(m):
        bit_i = 1 << i
        for mask in range(count):
            if not mask & bit_i:
                size = mask.bit_count()
                phi[:, i] += (values[mask | bit_i] - values[mask]) / (m * comb(m - 1, size))
        for j in range(i + 1, m):
            bit_j = 1 << j
            for mask in range(count):
                if not mask & (bit_i | bit_j):
                    size = mask.bit_count()
                    delta = (values[mask | bit_i | bit_j] - values[mask | bit_i]
                             - values[mask | bit_j] + values[mask])
                    interaction[:, i, j] += delta / (2 * (m - 1) * comb(m - 2, size))
            interaction[:, j, i] = interaction[:, i, j]
    for i in range(m):
        interaction[:, i, i] = phi[:, i] - interaction[:, i, :].sum(dim=1)
    residual = phi.sum(dim=1) - (scores[:, -1] - scores[:, 0])
    if not torch.isfinite(phi).all() or not torch.isfinite(interaction).all() or not torch.isfinite(residual).all():
        raise ValueError("attribution arithmetic overflowed; rescale the score objective")
    return ModalityShapleyResult(names, scores, phi, interaction, residual)


def modality_ablations(scorer, inputs, references, *, max_evaluations=1_000_000):
    """Full-minus-without and only-minus-reference effects (not Shapley values)."""
    _scorer(scorer)
    inputs = _inputs(inputs)
    references = _references(inputs, references)
    names, m = tuple(inputs), len(inputs)
    full = (1 << m) - 1
    masks = [0, full] + [full ^ (1 << i) for i in range(m)] + [1 << i for i in range(m)]
    values = _game(scorer, inputs, references, masks, max_evaluations)
    without = torch.stack([values[full ^ (1 << i)] for i in range(m)], dim=1)
    only = torch.stack([values[1 << i] for i in range(m)], dim=1)
    removal, addition = values[full][:, None] - without, only - values[0][:, None]
    if not torch.isfinite(removal).all() or not torch.isfinite(addition).all():
        raise ValueError("ablation arithmetic overflowed; rescale the score objective")
    return ModalityAblationResult(names, values[full], values[0], without, only, removal, addition)


def mismatched_pairs(scorer, inputs, *, modality, permutation=None, max_evaluations=1_000_000):
    """Replace one modality by a different row; default is a deterministic cyclic shift.

    A supplied permutation must be an integer bijection with no fixed points.
    This descriptive pairing diagnostic is not a causal or significance test.
    """
    _scorer(scorer)
    _positive(max_evaluations, "max_evaluations")
    inputs = _inputs(inputs)
    if modality not in inputs:
        raise ValueError("modality must name an input")
    value = inputs[modality]
    batch = value.shape[0]
    if batch < 2:
        raise ValueError("mismatched pairs require at least two observations")
    if 2 * batch > max_evaluations:
        raise ValueError("requested pairing rows exceed max_evaluations")
    rows = torch.arange(batch, device=value.device)
    if permutation is None:
        permutation = rows.roll(1)
    if (not isinstance(permutation, torch.Tensor) or permutation.shape != (batch,)
            or permutation.dtype not in (torch.int32, torch.int64) or permutation.device != value.device):
        raise ValueError("permutation must be integer [batch] on the input device")
    permutation = permutation.detach().to(torch.long).clone()
    if not torch.equal(permutation.sort().values, rows) or (permutation == rows).any():
        raise ValueError("permutation must be a bijection with no fixed points")
    matched = scorer._score(inputs)
    changed = dict(inputs)
    changed[modality] = value[permutation]
    mismatched = scorer._score(changed)
    drop = matched - mismatched
    if not torch.isfinite(drop).all():
        raise ValueError("pairing arithmetic overflowed; rescale the score objective")
    return MismatchedPairResult(modality, permutation, matched, mismatched, drop)


def emap(scorer, inputs, *, batch_size=128, max_pairs=65_536) -> EMAPResult:
    """Two-modality empirical additive projection over all N*N Cartesian pairs.

    For F[i,j]=score(x_i,y_j), projection[i,j] is row_mean[i] +
    column_mean[j] - grand_mean. Scoring must be independent across batch rows.
    """
    _scorer(scorer)
    _positive(batch_size, "batch_size")
    _positive(max_pairs, "max_pairs")
    inputs = _inputs(inputs)
    if len(inputs) != 2:
        raise ValueError("EMAP requires exactly two named modalities")
    names = tuple(inputs)
    first, second = (inputs[name] for name in names)
    n = first.shape[0]
    if n * n > max_pairs:
        raise ValueError("Cartesian EMAP exceeds max_pairs")
    chunks = []
    for start in range(0, n * n, batch_size):
        flat = torch.arange(start, min(start + batch_size, n * n), device=first.device)
        chunks.append(scorer._score({names[0]: first[flat // n], names[1]: second[flat % n]}))
    pairs = torch.cat(chunks).reshape(n, n)
    row, column, grand = pairs.mean(dim=1), pairs.mean(dim=0), pairs.mean()
    additive = row[:, None] + column[None, :] - grand
    residual = pairs - additive
    first_effect, second_effect = row - grand, column - grand
    if not all(torch.isfinite(value).all() for value in (grand, additive, residual, first_effect, second_effect)):
        raise ValueError("projection arithmetic overflowed; rescale the score objective")
    return EMAPResult(names, pairs, grand, first_effect, second_effect,
                      additive, residual, pairs.diagonal().clone(), additive.diagonal().clone())


def compare_answer_explanation_reliance(answer_scorer, explanation_scorer, inputs, references,
                                        *, max_coalitions=256, max_evaluations=1_000_000):
    """Compare normalized absolute modality Shapley values of two fixed objectives.

    max_evaluations bounds combined scored rows across both games. This compares
    score sensitivities, not explanation truthfulness or causal faithfulness.
    """
    _positive(max_evaluations, "max_evaluations")
    _scorer(answer_scorer)
    _scorer(explanation_scorer)
    inputs = _inputs(inputs)
    references = _references(inputs, references)
    required = 2 * (1 << len(inputs)) * next(iter(inputs.values())).shape[0]
    if required > max_evaluations:
        raise ValueError("both reliance games exceed max_evaluations")
    answer = modality_shapley(answer_scorer, inputs, references, max_coalitions=max_coalitions,
                             max_evaluations=max_evaluations // 2)
    explanation = modality_shapley(explanation_scorer, inputs, references, max_coalitions=max_coalitions,
                                  max_evaluations=max_evaluations // 2)

    def shares(values):
        # Rescale before summing to avoid overflow in absolute totals.
        absolute = values.abs()
        scale = absolute.amax(dim=1, keepdim=True)
        valid = scale[:, 0] > 0
        scaled = absolute / torch.where(scale > 0, scale, torch.ones_like(scale))
        total = scaled.sum(dim=1, keepdim=True)
        return scaled / torch.where(total > 0, total, torch.ones_like(total)), valid

    answer_shares, answer_valid = shares(answer.contributions)
    explanation_shares, explanation_valid = shares(explanation.contributions)
    valid = answer_valid & explanation_valid
    distance = (answer_shares - explanation_shares).abs().sum(dim=1)
    distance = torch.where(valid, distance, torch.full_like(distance, float("nan")))
    return RelianceComparison(answer, explanation, answer_shares, explanation_shares, valid, distance)
