"""Bounded donor patching and descriptive cross-modal attention inspection.

Architecture-independent protocols, not complete implementations of particular
paper methods. Adapters must expose explicit, aligned tensor-output sites.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
import math

import torch

from .core import evaluating


@dataclass(frozen=True)
class PatchSite:
    """One replacement site; mask broadcasts to the FULL activation including B.

    alignment is a required human-readable contract explaining why clean and
    corrupt positions represent corresponding tokens/regions/frames. Equal
    tensor shapes alone cannot establish semantic alignment. mask=None replaces
    the entire activation; otherwise a nonempty boolean mask is required.
    """
    name: str
    layer: str
    alignment: str
    mask: torch.Tensor | None = None


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def donor_patching_sweep(model, clean_inputs, corrupt_inputs, *, sites, score_fn,
                         max_sites=128, max_activation_elements=16_000_000,
                         max_forwards=1024, recovery_epsilon=1e-8):
    """Clean -> corrupt restoration with clean/corrupt self-patch controls.

    Both input mappings contain named batched tensors accepted by model(**kw).
    score_fn returns finite [B] scores for the SAME explicit chosen output on
    every run (e.g. teacher-forced answer log probability, not reselected argmax).
    Each site is independently evaluated; no cumulative patches or cache reuse.
    Outputs are raw CPU float64 scores plus recovery
    (patched_corrupt - corrupt)/(clean - corrupt), not clipped to [0,1].
    Near-zero denominators produce NaN recovery and recovery_valid=False.
    Nonfinite activations, computed differences, or otherwise-valid recovery
    values raise ValueError rather than silently returning overflow.

    A selected module MUST execute exactly once and return a floating tensor.
    Donors/recipients must have identical shape, dtype and device. Structured
    outputs, recurrent/shared sites, generation/KV caches, token realignment,
    distributed/offloaded layers and fused internal sites require user adapters.
    Calls must not overlap other hooks, concurrent forwards or model mutation.
    Hooks are removed on every exit; heterogeneous training flags are restored.
    no_grad is used and parameter gradients are neither cleared nor accumulated.

    At most two detached activation clones (each <= max_activation_elements)
    are retained between forwards. Masks and caller/model allocations are not
    included in this bound. Baselines and capture passes are explicit forwards.
    """
    for value, name in ((max_sites, "max_sites"), (max_activation_elements, "max_activation_elements"),
                        (max_forwards, "max_forwards")):
        _positive_int(value, name)
    if not math.isfinite(recovery_epsilon) or recovery_epsilon <= 0:
        raise ValueError("recovery_epsilon must be finite and positive")
    if not isinstance(clean_inputs, Mapping) or not isinstance(corrupt_inputs, Mapping):
        raise ValueError("clean and corrupt inputs must be named tensor mappings")
    if not clean_inputs or set(clean_inputs) != set(corrupt_inputs):
        raise ValueError("clean and corrupt mappings must have identical nonempty keys")
    values = list(clean_inputs.values()) + list(corrupt_inputs.values())
    if any(not isinstance(v, torch.Tensor) or v.ndim < 1 for v in values):
        raise ValueError("all model inputs must be batched tensors")
    batch = values[0].shape[0]
    if not batch or any(v.shape[0] != batch for v in values):
        raise ValueError("all model inputs must share a nonempty batch")
    sites = tuple(islice(sites, max_sites + 1))
    if any(not isinstance(site, PatchSite) for site in sites):
        raise ValueError("sites must contain PatchSite instances")
    if any(not isinstance(site.name, str) or not site.name for site in sites):
        raise ValueError("site names must be nonempty strings")
    if not 0 < len(sites) <= max_sites or len({site.name for site in sites}) != len(sites):
        raise ValueError("sites must have unique names and fit max_sites")
    if 2 + 5 * len(sites) > max_forwards:
        raise ValueError("sweep exceeds max_forwards")
    modules = dict(model.named_modules())
    for site in sites:
        if (not isinstance(site.layer, str) or site.layer not in modules
                or not isinstance(site.alignment, str) or not site.alignment.strip()):
            raise ValueError("each site needs a known layer and explicit alignment description")
        if site.mask is not None and (not isinstance(site.mask, torch.Tensor)
                                     or site.mask.layout != torch.strided or site.mask.is_nested
                                     or site.mask.device.type == "meta" or site.mask.dtype != torch.bool):
            raise ValueError("site masks must be dense boolean tensors")

    def run(inputs):
        output = model(**{key: value.detach().clone() for key, value in inputs.items()})
        score = score_fn(output)
        if (not isinstance(score, torch.Tensor) or score.shape != (batch,)
                or not score.is_floating_point() or not torch.isfinite(score).all()):
            raise ValueError("score_fn must return finite floating scores [B]")
        return score.detach().to(device="cpu", dtype=torch.float64).clone()

    def hooked(inputs, site, replacement=None):
        captured = []
        calls = 0

        def hook(module, args, output):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise ValueError("selected module must execute exactly once per forward")
            if (not isinstance(output, torch.Tensor) or output.layout != torch.strided
                    or output.is_nested or output.device.type == "meta" or not output.is_floating_point()):
                raise ValueError("selected module must return a dense floating tensor, not a structured output")
            if output.numel() > max_activation_elements:
                raise ValueError("activation exceeds max_activation_elements")
            if not torch.isfinite(output).all():
                raise ValueError("captured and recipient activations must be finite")
            if replacement is None:
                captured.append(output.detach().clone())
                return None
            if (replacement.shape != output.shape or replacement.dtype != output.dtype
                    or replacement.device != output.device):
                raise ValueError("clean/corrupt activations must match shape, dtype and device exactly")
            if site.mask is None:
                return replacement.clone()
            try:
                selected = torch.broadcast_to(site.mask.detach().to(output.device), output.shape)
            except RuntimeError as exc:
                raise ValueError("site mask must broadcast to full activation shape including batch") from exc
            if not selected.any():
                raise ValueError("site mask must select at least one activation element")
            return torch.where(selected, replacement, output)

        handle = modules[site.layer].register_forward_hook(hook)
        try:
            score = run(inputs)
            if calls != 1:
                raise ValueError("selected module did not execute")
            return score, captured[0] if captured else None
        finally:
            handle.remove()

    rows = []
    with evaluating(model), torch.no_grad():
        clean_score, corrupt_score = run(clean_inputs), run(corrupt_inputs)
        denominator = clean_score - corrupt_score
        if not torch.isfinite(denominator).all():
            raise ValueError("clean-minus-corrupt recovery denominator overflowed")
        valid = denominator.abs() > recovery_epsilon
        for site in sites:
            clean_capture_score, donor = hooked(clean_inputs, site)
            corrupt_capture_score, recipient = hooked(corrupt_inputs, site)
            if donor.shape != recipient.shape or donor.dtype != recipient.dtype or donor.device != recipient.device:
                raise ValueError("clean and corrupt activations are not shape/device/dtype aligned")
            clean_self, _ = hooked(clean_inputs, site, donor)
            corrupt_self, _ = hooked(corrupt_inputs, site, recipient)
            patched, _ = hooked(corrupt_inputs, site, donor)
            clean_self_delta = clean_self - clean_capture_score
            corrupt_self_delta = corrupt_self - corrupt_capture_score
            restoration_delta = patched - corrupt_score
            if any(not torch.isfinite(delta).all()
                   for delta in (clean_self_delta, corrupt_self_delta, restoration_delta)):
                raise ValueError("self-patch or restoration score difference overflowed")
            recovery = torch.full_like(denominator, math.nan)
            recovery[valid] = restoration_delta[valid] / denominator[valid]
            if not torch.isfinite(recovery[valid]).all():
                raise ValueError("recovery overflowed for a nonzero denominator")
            rows.append({"site": site.name, "layer": site.layer, "alignment": site.alignment,
                         "activation_shape": tuple(donor.shape),
                         "clean_capture_scores": clean_capture_score,
                         "corrupt_capture_scores": corrupt_capture_score,
                         "clean_self_scores": clean_self, "corrupt_self_scores": corrupt_self,
                         "clean_self_delta": clean_self_delta,
                         "corrupt_self_delta": corrupt_self_delta,
                         "patched_corrupt_scores": patched, "restoration_delta": restoration_delta,
                         "recovery": recovery, "recovery_valid": valid.clone()})
            del donor, recipient
    return {"clean_scores": clean_score, "corrupt_scores": corrupt_score,
            "recovery_denominator": denominator, "sites": rows,
            "interpretation": "controlled_activation_intervention_not_automatic_semantic_causality"}


def cross_modal_attention_payload(attentions, *, query_modalities, key_modalities,
                                  include_matrices=False, max_layers=32,
                                  max_attention_elements=16_000_000,
                                  max_payload_elements=100_000):
    """Serializable, descriptive summary of explicit [B,H,Q,K] attention tensors.

    attentions is an iterable of (layer_name, tensor) pairs. Supply a generator
    when possible: only one layer is processed at a time and no tensor is retained
    in the result. This cannot prevent the upstream model from materializing all
    attention layers; disable caches and arrange streamed extraction in its adapter.
    Each query/key modality maps to unique explicit sequence indices (including
    any offset for prefix/image expansion); modalities on each axis cannot overlap.
    Non-square cross-attention is supported. Indices absent from the mappings are
    intentionally unreported (e.g. padding/special tokens), not renormalized away.

    Values must be finite nonnegative attention probabilities with full row sums
    <= 1 + 1e-4 (zero rows for masked queries are allowed). Run extraction in eval
    mode and supply pre-dropout probabilities, not logits. These numerical checks
    alone cannot certify how a caller's weights were obtained.
    Summary mean_key_mass [B,H] averages total selected-key attention per selected
    query; mean_pair_weight divides again by the number of selected keys. Optional
    matrices [B,H,Q_selected,K_selected] are bounded across the whole payload.
    These weights are descriptive routing observations, NOT causal attribution.
    Caps bound accepted input elements and serialized numerical elements, excluding
    string metadata and input tensors already allocated by the caller/model.
    """
    for value, name in ((max_layers, "max_layers"), (max_attention_elements, "max_attention_elements"),
                        (max_payload_elements, "max_payload_elements")):
        _positive_int(value, name)

    def normalize(mapping):
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError("query/key modalities must be nonempty mappings")
        result, seen = {}, set()
        index_count = 0
        for name, indices in mapping.items():
            if not isinstance(name, str) or not name:
                raise ValueError("modality names must be nonempty strings")
            indices = list(islice(indices, max_payload_elements - index_count + 1))
            index_count += len(indices)
            if index_count > max_payload_elements:
                raise ValueError("token-index metadata exceeds max_payload_elements")
            if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError("modality indices must be nonempty sequences of nonnegative Python integers")
            if len(set(indices)) != len(indices) or seen.intersection(indices):
                raise ValueError("modality indices must be unique and disjoint on each axis")
            seen.update(indices)
            result[name] = indices
        return result

    queries, keys = normalize(query_modalities), normalize(key_modalities)
    consumed = sum(map(len, queries.values())) + sum(map(len, keys.values()))
    if consumed > max_payload_elements:
        raise ValueError("token-index metadata exceeds max_payload_elements")
    result = {"interpretation": "descriptive_attention_not_causal_attribution",
              "query_modalities": queries, "key_modalities": keys, "layers": []}
    names = set()
    for layer_number, (name, attention) in enumerate(attentions):
        if layer_number >= max_layers:
            raise ValueError("attention iterable exceeds max_layers")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("layer names must be nonempty and unique")
        names.add(name)
        if (not isinstance(attention, torch.Tensor) or attention.ndim != 4
                or not attention.is_floating_point() or any(size == 0 for size in attention.shape)):
            raise ValueError("attention must be floating [B,H,Q,K] with nonempty dimensions")
        if attention.numel() > max_attention_elements:
            raise ValueError("attention exceeds max_attention_elements")
        attention = attention.detach()
        if (not torch.isfinite(attention).all() or (attention < 0).any()
                or (attention.sum(-1, dtype=torch.float64) > 1.0001).any()):
            raise ValueError("attention must contain nonnegative finite probabilities with row sums <= 1")
        if any(max(indices) >= attention.shape[2] for indices in queries.values()):
            raise ValueError("query indices exceed query sequence length")
        if any(max(indices) >= attention.shape[3] for indices in keys.values()):
            raise ValueError("key indices exceed key sequence length")
        layer = {"name": name, "shape": list(attention.shape), "pairs": []}
        consumed += 4
        for query_name, query_indices in queries.items():
            for key_name, key_indices in keys.items():
                cells = attention.shape[0] * attention.shape[1]
                needed = 2 * cells + (cells * len(query_indices) * len(key_indices) if include_matrices else 0)
                if consumed + needed > max_payload_elements:
                    raise ValueError("attention summaries/matrices exceed max_payload_elements")
                consumed += needed
                q = torch.tensor(query_indices, device=attention.device)
                k = torch.tensor(key_indices, device=attention.device)
                # Advanced indexing selects only this block, not a full Q x K copy.
                block = attention[:, :, q[:, None], k[None, :]].to(device="cpu", dtype=torch.float64)
                mass = block.sum(-1).mean(-1)
                pair = {"query_modality": query_name, "key_modality": key_name,
                        "mean_key_mass": mass.tolist(),
                        "mean_pair_weight": (mass / len(key_indices)).tolist()}
                if include_matrices:
                    pair["matrix"] = block.tolist()
                layer["pairs"].append(pair)
                del block
        result["layers"].append(layer)
    if not result["layers"]:
        raise ValueError("at least one attention layer is required")
    return result
