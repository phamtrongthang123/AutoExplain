"""Explicit grouped interventions, grounding overlap and fixed-output LM scores.

These protocols do not infer modality layouts, choose an answer, or declare that
annotation overlap establishes faithfulness. Model adapters own those decisions.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
import math

import torch

from .core import evaluating


@dataclass(frozen=True)
class EvidenceGroup:
    """Named boolean mask over one modality's non-batch dimensions.

    Masks may broadcast to inputs[modality].shape[1:] (e.g. [T,1] token
    embeddings or [1,H,W] image regions). Groups must be disjoint within a
    modality. Layouts need not be square, contiguous, or image-shaped.
    """
    name: str
    modality: str
    mask: torch.Tensor


@dataclass(frozen=True)
class GroupedEvidenceResult:
    budgets: torch.Tensor
    fractions: torch.Tensor
    group_order: tuple[str, ...]
    original_scores: torch.Tensor
    replacement_scores: torch.Tensor
    deletion_scores: torch.Tensor
    insertion_scores: torch.Tensor
    sufficiency_gap: torch.Tensor
    necessity_drop: torch.Tensor
    deletion_auc: torch.Tensor
    insertion_auc: torch.Tensor
    random_orders: tuple[tuple[str, ...], ...]
    random_deletion_scores: torch.Tensor
    random_insertion_scores: torch.Tensor
    random_sufficiency_gap: torch.Tensor
    random_necessity_drop: torch.Tensor
    random_deletion_auc: torch.Tensor
    random_insertion_auc: torch.Tensor


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _score_vector(output, score_fn, batch):
    score = score_fn(output)
    if not isinstance(score, torch.Tensor) or score.shape != (batch,):
        raise ValueError("score_fn must return a tensor [batch] for a fixed explicit output")
    if not score.is_floating_point() or not torch.isfinite(score).all():
        raise ValueError("score_fn must return finite floating scores")
    return score.detach().to(device="cpu", dtype=torch.float64).clone()


def grouped_evidence_evaluation(model, inputs: Mapping[str, torch.Tensor], *,
                                groups: Sequence[EvidenceGroup], replacements,
                                importance, score_fn: Callable,
                                budgets=None, random_trials=5, generator=None,
                                max_groups=2048, max_forwards=10000):
    """Deletion/insertion and keep-only sufficiency/remove-only necessity.

    model(**inputs) must accept batched named tensor inputs. score_fn selects the
    SAME explicit output for every forward and returns [B]; no argmax retargeting
    is performed. Importance is one finite signed score per group, ranked high
    to low (take abs explicitly if desired). Budgets count groups, not pixels.

    Replacement values must be explicit full-shape tensors for every intervened
    modality, with matching device/dtype. Uncovered elements and other modalities
    remain original even at the insertion baseline. Random orders preserve each
    ranked slot's modality AND expanded group size, making controls matched in
    group count and replaced scalar count at every budget. Singleton strata
    necessarily produce identical controls. A CPU or device-local generator is
    supported without touching its seed or global RNG when supplied.

    All scores are CPU float64. Curves have shape [B,K], controls [R,B,K].
    Sufficiency gap = original - keep-only; necessity drop = original - deletion.
    These are raw score differences, not calibrated probabilities or causal
    claims. AUC uses group fractions over the supplied budget range only.
    """
    _positive_int(random_trials, "random_trials")
    _positive_int(max_groups, "max_groups")
    _positive_int(max_forwards, "max_forwards")
    if not isinstance(inputs, Mapping) or not inputs or any(
            not isinstance(v, torch.Tensor) or v.layout != torch.strided or v.is_nested
            or v.is_quantized or v.is_complex() or v.device.type == "meta"
            or v.ndim < 1 or v.numel() == 0 for v in inputs.values()):
        raise ValueError("inputs must be a nonempty mapping of real dense nonempty batched tensors")
    if any(not isinstance(key, str) or not key for key in inputs):
        raise ValueError("input names must be nonempty strings")
    if any(not torch.isfinite(value).all() for value in inputs.values()):
        raise ValueError("all inputs, including nonintervened inputs, must be finite")
    batch = next(iter(inputs.values())).shape[0]
    if batch == 0 or any(v.shape[0] != batch for v in inputs.values()):
        raise ValueError("all inputs must have the same nonempty batch")
    groups = tuple(islice(groups, max_groups + 1))
    count = len(groups)
    if any(not isinstance(group, EvidenceGroup) for group in groups):
        raise ValueError("groups must contain EvidenceGroup instances")
    if any(not isinstance(g.name, str) or not g.name
           or not isinstance(g.modality, str) or not g.modality for g in groups):
        raise ValueError("group names and modalities must be nonempty strings")
    if not 0 < count <= max_groups or len({g.name for g in groups}) != count:
        raise ValueError("groups must have unique names and fit max_groups")
    importance = torch.as_tensor(importance).detach().cpu()
    if importance.shape != (count,) or importance.is_complex() or not torch.isfinite(importance).all():
        raise ValueError("importance must contain one finite value per group")
    if budgets is None:
        budgets = list(range(count + 1))
    else:
        budgets = list(islice(budgets, count + 2))
    if (not budgets or any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= count
                           for k in budgets) or budgets != sorted(set(budgets))):
        raise ValueError("budgets must be strictly increasing integer group counts in [0,G]")
    if 2 + 2 * len(budgets) * (1 + random_trials) > max_forwards:
        raise ValueError("requested curves exceed max_forwards")
    originals = {key: value.detach().clone() for key, value in inputs.items()}
    modalities = {g.modality for g in groups}
    if not isinstance(replacements, Mapping) or set(replacements) != modalities:
        raise ValueError("replacements must name exactly the intervened modalities")
    bases, covered, masks, strata = {}, {}, [], {}
    for key in modalities:
        if key not in originals:
            raise ValueError(f"unknown input modality {key!r}")
        x, replacement = originals[key], replacements[key]
        if (not isinstance(replacement, torch.Tensor) or replacement.layout != torch.strided
                or replacement.is_nested or replacement.is_quantized or replacement.is_complex()
                or replacement.device.type == "meta" or replacement.numel() == 0
                or replacement.shape != x.shape or replacement.dtype != x.dtype or replacement.device != x.device):
            raise ValueError("replacement must be real dense nonempty and match input shape, dtype and device")
        if not torch.isfinite(replacement).all():
            raise ValueError("replacements must be finite")
        bases[key] = replacement.detach().clone()
        covered[key] = torch.zeros(x.shape[1:], dtype=torch.bool, device=x.device)
    for index, group in enumerate(groups):
        x = originals[group.modality]
        if (not isinstance(group.mask, torch.Tensor) or group.mask.layout != torch.strided
                or group.mask.is_nested or group.mask.device.type == "meta" or group.mask.dtype != torch.bool):
            raise ValueError("group masks must be dense boolean tensors")
        try:
            mask = torch.broadcast_to(group.mask.detach().to(x.device), x.shape[1:]).clone()
        except RuntimeError as exc:
            raise ValueError("group mask must broadcast to non-batch input shape") from exc
        size = int(mask.sum().item())
        if not size or (covered[group.modality] & mask).any():
            raise ValueError("group masks must be nonempty and disjoint within each modality")
        covered[group.modality] |= mask
        masks.append(mask)
        strata.setdefault((group.modality, size), []).append(index)
    ranking = importance.argsort(descending=True, stable=True).tolist()
    orders = []
    random_device = generator.device if generator is not None else torch.device("cpu")
    for _ in range(random_trials):
        mapping = {}
        for members in strata.values():
            permutation = torch.randperm(len(members), generator=generator, device=random_device).cpu().tolist()
            mapping.update({member: members[p] for member, p in zip(members, permutation)})
        orders.append([mapping[index] for index in ranking])

    def evaluate(values):
        # Protect stored inputs even if a user model mutates its arguments.
        return _score_vector(model(**{k: v.clone() for k, v in values.items()}), score_fn, batch)

    baseline = {key: (torch.where(covered[key], bases[key], value) if key in modalities else value)
                for key, value in originals.items()}

    def curves(order):
        deletion, insertion = [], []
        for budget in budgets:
            selected = {key: torch.zeros_like(mask) for key, mask in covered.items()}
            for index in order[:budget]:
                selected[groups[index].modality] |= masks[index]
            removed = dict(originals)
            retained = dict(baseline)
            for key, mask in selected.items():
                removed[key] = torch.where(mask, bases[key], originals[key])
                retained[key] = torch.where(mask, originals[key], baseline[key])
            deletion.append(evaluate(removed))
            insertion.append(evaluate(retained))
        return torch.stack(deletion, dim=1), torch.stack(insertion, dim=1)

    with evaluating(model), torch.no_grad():
        original_score = evaluate(originals)
        replacement_score = evaluate(baseline)
        deletion, insertion = curves(ranking)
        controls = [curves(order) for order in orders]
    random_deletion = torch.stack([value[0] for value in controls])
    random_insertion = torch.stack([value[1] for value in controls])
    fractions = torch.tensor(budgets, dtype=torch.float64) / count
    return GroupedEvidenceResult(
        torch.tensor(budgets), fractions, tuple(groups[i].name for i in ranking),
        original_score, replacement_score, deletion, insertion,
        original_score[:, None] - insertion, original_score[:, None] - deletion,
        torch.trapezoid(deletion, fractions, dim=-1), torch.trapezoid(insertion, fractions, dim=-1),
        tuple(tuple(groups[i].name for i in order) for order in orders), random_deletion, random_insertion,
        original_score[None, :, None] - random_insertion,
        original_score[None, :, None] - random_deletion,
        torch.trapezoid(random_deletion, fractions, dim=-1),
        torch.trapezoid(random_insertion, fractions, dim=-1),
    )


def grounded_evidence_metrics(predicted, reference, *, semantics, weights=None):
    """Per-sample overlap of explicit boolean [B,...] annotation masks.

    semantics must be 'spatial', 'temporal', or 'spatiotemporal'; this is caller
    metadata, never inferred. Both masks must use the same coordinates/support.
    Temporal bins are half-open [edge_i, edge_{i+1}); provide duration weights for
    unequal bins. Spatial weights may encode pixel/region area, and joint weights
    space-time volume. No automatic rasterization, resizing or thresholding.

    Empty/empty: precision=recall=IoU=Dice=1. Empty prediction against nonempty
    reference: precision=1, recall=0; converse: precision=0, recall=1. IoU/Dice
    are zero for one-sided emptiness. Zero-weight support is treated as empty.
    This measures annotation plausibility, NOT model faithfulness.
    """
    if semantics not in {"spatial", "temporal", "spatiotemporal"}:
        raise ValueError("explicit spatial/temporal/spatiotemporal semantics required")
    if (not isinstance(predicted, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or predicted.dtype != torch.bool or reference.dtype != torch.bool
            or predicted.shape != reference.shape or predicted.ndim < 2 or predicted.shape[0] == 0):
        raise ValueError("predicted and reference must be same-shape boolean [B,...] masks")
    predicted = predicted.detach().cpu()
    reference = reference.detach().cpu()
    if weights is None:
        weights = torch.ones(predicted.shape, dtype=torch.float64)
    else:
        weights = torch.as_tensor(weights).detach().to(device="cpu", dtype=torch.float64)
        try:
            weights = torch.broadcast_to(weights, predicted.shape)
        except RuntimeError as exc:
            raise ValueError("weights must broadcast to mask shape") from exc
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and nonnegative")
    mass = lambda mask: (mask * weights).flatten(1).sum(1)
    intersection, pred_mass, ref_mass = mass(predicted & reference), mass(predicted), mass(reference)
    union = mass(predicted | reference)
    if not torch.isfinite(pred_mass + ref_mass).all():
        raise ValueError("weighted mask mass overflowed; rescale weights")

    def ratio(numerator, denominator):
        positive = denominator > 0
        safe = torch.where(positive, denominator, torch.ones_like(denominator))
        return torch.where(positive, numerator / safe, torch.ones_like(denominator))

    return {"semantics": semantics, "interpretation": "annotation_overlap_not_faithfulness",
            "precision": ratio(intersection, pred_mass), "recall": ratio(intersection, ref_mass),
            "iou": ratio(intersection, union), "dice": ratio(2 * intersection, pred_mass + ref_mass),
            "intersection": intersection, "prediction_mass": pred_mass, "reference_mass": ref_mass,
            "prediction_empty": pred_mass == 0, "reference_empty": ref_mass == 0}


def fixed_prefix_log_probs(logits, chosen_ids, *, attention_mask, output_mask):
    """Teacher-forced causal scores for explicit chosen tokens in a fixed sequence.

    logits [B,T,V] correspond to chosen_ids [B,T] fed to the model. Logits at
    position t-1 score chosen_ids[:,t]. output_mask [B,T] selects ONLY the chosen
    continuation/output tokens, excluding prompt tokens; attention_mask marks
    nonpadding positions. Both masks must be boolean. Selected tokens require a
    valid predecessor; position zero cannot be scored. Padding gaps may exist but
    selected tokens immediately after gaps are rejected. The caller must supply
    the SAME prefix and chosen continuation when comparing interventions.

    Returns [B,T] token scores (zero outside selection), [B] sum/mean/count.
    Empty outputs have sum=0 and mean=NaN with valid=False, never a silent mean 0.
    This function accepts logits, not model-specific output objects; adapters must
    explicitly select .logits/dict/tuple fields and align multimodal positions.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point()
            or logits.shape[0] == 0 or logits.shape[1] < 1 or logits.shape[2] < 1):
        raise ValueError("logits must be floating [B,T,V] with nonempty dimensions")
    shape = logits.shape[:2]
    if (not isinstance(chosen_ids, torch.Tensor) or chosen_ids.shape != shape
            or chosen_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("chosen_ids must be integer [B,T]")
    if any(not isinstance(mask, torch.Tensor) or mask.shape != shape or mask.dtype != torch.bool
           for mask in (attention_mask, output_mask)):
        raise ValueError("attention_mask and output_mask must be boolean [B,T]")
    ids = chosen_ids.to(logits.device)
    valid = attention_mask.to(logits.device)
    selected = output_mask.to(logits.device)
    predecessor = torch.zeros_like(valid)
    predecessor[:, 1:] = valid[:, :-1]
    if (selected & ~(valid & predecessor)).any():
        raise ValueError("every selected token needs itself and its causal predecessor unpadded")
    if ((ids[selected] < 0) | (ids[selected] >= logits.shape[-1])).any():
        raise ValueError("selected chosen token IDs are outside the vocabulary")
    positions = selected[:, 1:].nonzero(as_tuple=True)
    # Only selected rows are materialized; padded token IDs may use sentinel values.
    selected_logits = logits[:, :-1][positions]
    if selected_logits.dtype not in (torch.float32, torch.float64):
        selected_logits = selected_logits.float()
    if not torch.isfinite(selected_logits).all():
        raise ValueError("selected logits must be finite")
    chosen = ids[:, 1:][positions].long()
    values = selected_logits.log_softmax(-1).gather(1, chosen[:, None]).squeeze(1)
    token_scores = torch.zeros(shape, dtype=values.dtype, device=logits.device)
    token_scores[positions[0], positions[1] + 1] = values
    counts = selected.sum(1)
    total = token_scores.sum(1)
    mean = torch.where(counts > 0, total / counts.clamp_min(1), torch.full_like(total, math.nan))
    return {"token_log_probs": token_scores, "sequence_log_prob": total,
            "mean_log_prob": mean, "token_count": counts, "valid": counts > 0}
