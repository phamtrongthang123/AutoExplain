"""Differentiable CLIP scores and explicit tensor-level attribution primitives.

No downloads, model discovery, tokenizer assumptions, or named-paper substitutes.
Callers supply encoders and expose internal tensors in their actual forward graph.
"""
import math

import torch


__all__ = ["clip_similarity", "clip_scores", "encode_clip_scores",
           "grad_eclip", "token_gradient_attribution", "patch_attribution_map",
           "projected_component_scores", "attention_value_decomposition"]


def _floating(x, name, ndim=None):
    if (not isinstance(x, torch.Tensor) or not x.is_floating_point()
            or not x.numel() or not x.isfinite().all()
            or (ndim is not None and x.ndim != ndim)):
        raise ValueError(f"{name} must be a finite nonempty floating tensor"
                         + (f" with {ndim} axes" if ndim is not None else ""))
    # Promote low precision without breaking autograd.
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _unit(x, name):
    norm = x.norm(dim=-1, keepdim=True)
    if not norm.isfinite().all() or (norm <= torch.finfo(x.dtype).eps).any():
        raise ValueError(f"{name} contains a zero, overflowing or numerically degenerate vector")
    return x / norm


def clip_similarity(image_embeddings, text_embeddings, *, scale=1.0):
    """Return differentiable [images,texts] scaled cosine similarities.

    `scale` is the multiplier itself (not log_scale); 1 means cosine, while
    model.logit_scale.exp() gives CLIP logits. No softmax/contrastive loss is
    implicit. Zero embeddings are rejected, not silently normalized to zero.
    Embeddings must already share a projected latent space and device/dtype.
    Half/bfloat16 inputs are promoted to float32 for normalization and scoring.
    """
    image = _floating(image_embeddings, "image_embeddings", 2)
    text = _floating(text_embeddings, "text_embeddings", 2)
    if image.shape[1] != text.shape[1] or image.device != text.device or image.dtype != text.dtype:
        raise ValueError("Image/text embedding widths, devices and promoted dtypes must agree")
    scale = torch.as_tensor(scale, device=image.device, dtype=image.dtype)
    if scale.ndim != 0 or not scale.isfinite() or scale <= 0:
        raise ValueError("scale must be a finite positive scalar multiplier")
    return scale * (_unit(image, "image_embeddings") @ _unit(text, "text_embeddings").T)


def clip_scores(image_embeddings, text_embeddings, *, target=None, scale=1.0):
    """Select one text similarity per image, returning [images].

    target=None means aligned pairs and requires equal batch sizes. An integer
    selects one fixed text for all images; an integer [images] tensor selects
    independently. Scores are not probabilities and never infer an argmax target.
    """
    scores = clip_similarity(image_embeddings, text_embeddings, scale=scale)
    if target is None:
        if scores.shape[0] != scores.shape[1]:
            raise ValueError("Aligned-pair scoring requires equal image/text batch sizes")
        return scores.diagonal()
    indices = torch.as_tensor(target, device=scores.device)
    if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("target must contain integer text indices")
    if indices.ndim == 0:
        indices = indices.expand(scores.shape[0])
    if indices.shape != (scores.shape[0],) or (indices < 0).any() or (indices >= scores.shape[1]).any():
        raise ValueError("target must index an existing text for each image")
    return scores.gather(1, indices.long()[:, None]).squeeze(1)


def encode_clip_scores(images, texts, *, image_encoder, text_encoder, target=None, scale=1.0):
    """Score caller-provided differentiable encoders; preserve mode and graph.

    Encoders return [batch,embedding] projected embeddings. Caller supplies image
    preprocessing/tokenization, eval mode, and any attention masks. `texts` can
    be any input accepted by text_encoder (e.g. a closure over a token dictionary).
    This function neither freezes parameters nor changes their .grad buffers.
    """
    return clip_scores(image_encoder(images), text_encoder(texts), target=target, scale=scale)


def token_gradient_attribution(scores, tokens, *, retain_graph=False):
    """Signed gradient-times-token scores, NOT Grad-ECLIP or attention rollout.

    scores is [B], tokens is the actual differentiable [B,T,C] tensor used by
    the forward, not a post-hoc view created after scoring. Each score must be
    independent of other batch rows: a summed-score VJP is used. This supports
    either image patch or text embedding tensors (not integer token IDs).
    No ReLU/min-max rescaling hides negative or zero relevance.
    """
    _floating(scores, "scores", 1)
    _floating(tokens, "tokens", 3)
    if scores.shape[0] != tokens.shape[0] or not scores.requires_grad or not tokens.requires_grad:
        raise ValueError("Need aligned differentiable scores and tokens from the same forward graph")
    gradient, = torch.autograd.grad(scores.sum(), tokens, retain_graph=retain_graph)
    # Multiplication in float32 avoids half-precision overflow where possible.
    values = _floating(tokens.detach(), "tokens")
    gradients = _floating(gradient.detach(), "gradients")
    return {"relevance": (values * gradients).sum(-1), "gradients": gradients,
            "method": "signed_gradient_x_token"}


def grad_eclip(scores, attention_output, values, query, keys, *, query_index,
               attention_mode, retain_graph=False):
    """Grad-ECLIP Eqs. (7)-(8), one layer, from explicit forward intermediates.

    Source: https://proceedings.mlr.press/v235/zhao24p.html, pp. 4-5;
    official generate_emap.py grad_eclip / grad_eclip_text specify cosine and
    min-max spatial weights. This is not a full encoder adapter/reproduction.

    scores [B] must be cosine matching scores from the SAME single-head forward.
    attention_output is the actual graph ancestor [B,Q,C] = softmax(QK/sqrt(C))V
    BEFORE output projection/residual. values [B,T,C] are raw projected V, NOT
    W_O V. query [B,D] and keys [B,T,D] are q_out/k_out AFTER applying the
    attention output Linear (including bias), as in the official implementation.
    query_index selects the CLS/EOS row of attention_output (integer or [B]).
    The caller selects/reorders value and key tokens together: image spatial
    normalization excludes CLS; official text normalization includes all tokens
    and crops BOS/EOS/padding only after layer aggregation.

    attention_mode must explicitly be 'single_head': the paper/official code
    recomputes attention across the full channel width in one head. Native
    multi-head intermediates are not an equivalent reproduction and are refused
    by contract. This routine cannot inspect/validate the caller's encoder.
    Batch rows must be independent, since a summed-score VJP is used. Inputs
    must come from the same forward, not newly computed attention-output views.

    Returns detached channel/spatial weights, signed pre-ReLU contributions and
    nonnegative relevance [B,T]. Constant (within dtype epsilon) spatial cosine
    maps have undefined min-max normalization: return zero weights/relevance and
    spatial_weights_defined=False rather than official-code division by zero.
    Zero query/key norms are rejected. No final map rescaling/interpolation.
    """
    _floating(scores, "scores", 1)
    output = _floating(attention_output, "attention_output", 3)
    values = _floating(values, "values", 3).detach()
    query = _floating(query, "query", 2).detach()
    keys = _floating(keys, "keys", 3).detach()
    if attention_mode != "single_head":
        raise ValueError("Grad-ECLIP requires explicitly declared single_head forward intermediates")
    batch, tokens, channels = values.shape
    if (scores.shape != (batch,) or output.shape[0] != batch
            or output.shape[2] != channels or keys.shape[:2] != (batch, tokens)
            or query.shape != (batch, keys.shape[2])):
        raise ValueError("Scores, attention output, values, query and keys have incompatible shapes")
    if any(x.device != values.device or x.dtype != values.dtype for x in (output, query, keys)):
        raise ValueError("Intermediate devices and promoted dtypes must agree")
    if scores.device != values.device or not scores.requires_grad or not attention_output.requires_grad:
        raise ValueError("Need differentiable scores/output on the intermediates' device")
    indices = torch.as_tensor(query_index, device=values.device)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_index must contain integer CLS/EOS positions")
    if indices.ndim == 0:
        indices = indices.expand(batch)
    if indices.shape != (batch,) or (indices < 0).any() or (indices >= output.shape[1]).any():
        raise ValueError("query_index must select a valid attention-output row per sample")
    cosine = (_unit(query, "query")[:, None] * _unit(keys, "keys")).sum(-1)
    minimum = cosine.amin(-1, keepdim=True)
    span = cosine.amax(-1, keepdim=True) - minimum
    defined = span > torch.finfo(cosine.dtype).eps
    denominator = torch.where(defined, span, torch.ones_like(span))
    spatial_weights = torch.where(defined, (cosine - minimum) / denominator, torch.zeros_like(cosine))
    # Differentiate the original tensor, not the float-promoted validation view.
    gradient, = torch.autograd.grad(scores.sum(), attention_output, retain_graph=retain_graph)
    gradient = _floating(gradient.detach(), "attention-output gradient", 3)
    channel_weights = gradient[torch.arange(batch, device=values.device), indices.long()]
    signed = (values * channel_weights[:, None] * spatial_weights[..., None]).sum(-1)
    if not signed.isfinite().all():
        raise ValueError("Grad-ECLIP aggregation overflowed")
    return {"relevance": signed.relu(), "signed_relevance": signed,
            "channel_weights": channel_weights, "spatial_weights": spatial_weights,
            "query_key_cosine": cosine, "spatial_weights_defined": defined.squeeze(-1),
            "method": "grad_eclip_single_layer_single_head"}


def patch_attribution_map(relevance, *, grid_shape, patch_indices=None):
    """Map [B,T] relevance to an explicitly declared (rows,columns) patch grid.

    patch_indices explicitly selects/arranges patch positions when CLS/register
    tokens exist. Grid order is caller-declared row-major; no square-grid guess,
    CLS-token guess, interpolation, or image preprocessing inversion is made.
    """
    relevance = _floating(relevance, "relevance", 2)
    if (len(grid_shape) != 2 or any(type(n) is not int or n <= 0 for n in grid_shape)):
        raise ValueError("grid_shape must contain two positive integer dimensions")
    count = math.prod(grid_shape)
    if patch_indices is not None:
        indices = torch.as_tensor(patch_indices, device=relevance.device)
        if (indices.ndim != 1 or indices.numel() != count
                or indices.dtype not in (torch.int32, torch.int64)
                or (indices < 0).any() or (indices >= relevance.shape[1]).any()
                or indices.unique().numel() != count):
            raise ValueError("patch_indices must be distinct valid indices matching grid area")
        relevance = relevance[:, indices.long()]
    if relevance.shape[1] != count:
        raise ValueError("Token count must equal grid area; explicitly select patches to exclude special tokens")
    return relevance.reshape(relevance.shape[0], *grid_shape)


def projected_component_scores(components, reference, *, remainder=None, scale=1.0):
    """Exact additive cosine-score bookkeeping in a *shared projected space*.

    components [B,K,D] may represent caller-derived patches, heads, or text
    tokens; remainder [B,D] contains biases/residual paths. Sum them to recover
    the unnormalized embedding. Divide EVERY contribution by the SAME full
    embedding norm, not by individual component norms. reference [B,D] is the
    opposite-modality embedding. Returns signed contributions and their sum.
    This does not decompose nonlinear LayerNorm/MLP blocks or implement the
    ICLR 2024 CLIP text-based decomposition architecture adapters.
    """
    components = _floating(components, "components", 3)
    reference = _floating(reference, "reference", 2)
    shape = (components.shape[0], components.shape[2])
    if reference.shape != shape or reference.device != components.device or reference.dtype != components.dtype:
        raise ValueError("Reference must match batch, embedding width, device and promoted dtype")
    if remainder is None:
        remainder = components.new_zeros(shape)
    else:
        remainder = _floating(remainder, "remainder", 2)
        if remainder.shape != shape or remainder.device != components.device or remainder.dtype != components.dtype:
            raise ValueError("Remainder must match reference shape/device/dtype")
    full = components.sum(1) + remainder
    full_norm = full.norm(dim=-1, keepdim=True)
    _unit(full, "full embedding")
    direction = _unit(reference, "reference")
    scale = torch.as_tensor(scale, device=full.device, dtype=full.dtype)
    if scale.ndim != 0 or not scale.isfinite() or scale <= 0:
        raise ValueError("scale must be a finite positive scalar")
    contributions = scale * (components * direction[:, None]).sum(-1) / full_norm
    residual_score = scale * (remainder * direction).sum(-1) / full_norm.squeeze(-1)
    total = contributions.sum(-1) + residual_score
    cosine = scale * (full / full_norm * direction).sum(-1)
    return {"contributions": contributions, "remainder_score": residual_score,
            "total_score": total, "direct_score": cosine,
            "additivity_residual": total - cosine, "embedding": full}


def attention_value_decomposition(attention, values, output_weight, *, output_bias=None):
    """Exact linear attention-output head/token contributions for ONE query.

    attention [B,H,T] are actual weights (after softmax/dropout if applicable),
    values [B,H,T,Dh] are projected values, output_weight [Dout,H*Dh] follows
    torch.nn.Linear orientation. Returns [B,H,T,Dout] terms and [B,Dout] bias.
    Their sum is the selected query's attention output, NOT the whole block or
    CLIP embedding. Residuals, norms, MLPs and final projections stay explicit.
    Attention weights are held as supplied; this is not a causal ablation.
    """
    attention = _floating(attention, "attention", 3)
    values = _floating(values, "values", 4)
    weight = _floating(output_weight, "output_weight", 2)
    if attention.shape != values.shape[:3] or weight.shape[1] != values.shape[1] * values.shape[3]:
        raise ValueError("Attention/value/head projection shapes disagree")
    if any(x.device != values.device or x.dtype != values.dtype for x in (attention, weight)):
        raise ValueError("Tensor devices and promoted dtypes must agree")
    projected = torch.einsum("bhtd,ohd->bhto", values, weight.reshape(weight.shape[0], values.shape[1], values.shape[3]))
    contributions = attention[..., None] * projected
    if output_bias is None:
        bias = values.new_zeros((values.shape[0], weight.shape[0]))
    else:
        output_bias = _floating(output_bias, "output_bias", 1)
        if (output_bias.shape != (weight.shape[0],) or output_bias.device != values.device
                or output_bias.dtype != values.dtype):
            raise ValueError("Output bias must match projection width/device/dtype")
        bias = output_bias.expand(values.shape[0], -1)
    return {"contributions": contributions, "bias": bias,
            "output": contributions.sum((1, 2)) + bias}
