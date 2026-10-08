"""SpLiCE-style nonnegative sparse coding and multimodal feature diagnostics.

A transparent objective/solver variant, not an exact paper reproduction. All
vocabulary embeddings and calibration means are supplied locally by the caller.
"""
import math

import torch


__all__ = ["sparse_concept_decomposition", "reconstruct_concepts",
           "intervene_concepts", "modality_feature_diagnostics"]


def _matrix(x, name):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1
            or not x.is_floating_point() or not x.isfinite().all()):
        raise ValueError(f"{name} must be finite nonempty floating [observations,features]")
    return x.detach().float() if x.dtype in (torch.float16, torch.bfloat16) else x.detach()


def _compatible(a, b):
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("Devices and promoted dtypes must agree")


def _center(x, mean, name):
    if mean is None:
        return x, x.new_zeros(x.shape[1])
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).detach()
    if mean.shape != (x.shape[1],) or not mean.isfinite().all():
        raise ValueError(f"{name} must be a finite vector matching embedding width")
    return x - mean, mean


@torch.no_grad()
def sparse_concept_decomposition(embeddings, vocabulary_embeddings, *, penalty=0.01,
                                 embedding_mean=None, vocabulary_mean=None,
                                 normalize=True, max_iter=2000, tolerance=1e-5):
    """Solve min_{Z>=0} .5 ||X - Z D||_F^2 + penalty * ||Z||_1.

    X=[B,Dembedding], dictionary D=[concepts,Dembedding]. Independently subtract
    caller-provided modality means, then (by default) normalize every row.
    No hidden mean estimation or vocabulary construction occurs. Supply means
    estimated on disjoint calibration data; different means change coordinates
    and assume that centered image/text spaces remain aligned. This is a
    SpLiCE-inspired L1 variant using monotone proximal gradient (ISTA), not the
    authors' preprocessing pipeline, optimizer, or constrained-budget variant.

    Each row is solved independently with step 1/||D||_2^2. Diagnostics include
    objective, KKT/projected-gradient residual and per-row convergence flags;
    iteration exhaustion returns a nonconverged result, never claims success.
    No gradients are retained: this solver is for analysis, not SAE training.
    Half/bfloat16 are promoted to float32; otherwise dtype/device are retained.
    """
    x_raw = _matrix(embeddings, "embeddings")
    dictionary_raw = _matrix(vocabulary_embeddings, "vocabulary_embeddings")
    _compatible(x_raw, dictionary_raw)
    if x_raw.shape[1] != dictionary_raw.shape[1]:
        raise ValueError("Vocabulary and target embedding widths must agree")
    if (not math.isfinite(penalty) or penalty < 0 or not math.isfinite(tolerance)
            or tolerance <= 0 or type(max_iter) is not int or max_iter < 1
            or type(normalize) is not bool):
        raise ValueError("Need nonnegative finite penalty, positive tolerance/iterations and bool normalize")
    x, target_mean = _center(x_raw, embedding_mean, "embedding_mean")
    dictionary, concept_mean = _center(dictionary_raw, vocabulary_mean, "vocabulary_mean")
    target_norm = x.norm(dim=1, keepdim=True)
    dictionary_norm = dictionary.norm(dim=1, keepdim=True)
    eps = torch.finfo(x.dtype).eps
    if not target_norm.isfinite().all() or not dictionary_norm.isfinite().all():
        raise ValueError("Centered embedding norms overflowed; rescale inputs or use float64")
    if (dictionary_norm <= eps).any():
        raise ValueError("Vocabulary contains zero/degenerate centered concepts; filter them explicitly")
    if normalize:
        if (target_norm <= eps).any():
            raise ValueError("Cannot normalize a zero/degenerate centered target")
        x = x / target_norm
        dictionary = dictionary / dictionary_norm
    # The spectral norm is computed once, not a guessed step or dense C-by-C Gram.
    lipschitz = torch.linalg.matrix_norm(dictionary, ord=2).square()
    if not lipschitz.isfinite() or lipschitz <= 0:
        raise ValueError("Dictionary has invalid spectral norm")
    codes = x.new_zeros((x.shape[0], dictionary.shape[0]))
    initial_gradient = -x @ dictionary.T + penalty
    threshold = tolerance * initial_gradient.abs().amax(1).clamp_min(1)
    if not initial_gradient.isfinite().all() or not threshold.isfinite().all():
        raise ValueError("Initial gradient or convergence threshold overflowed")
    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for iteration in range(1, max_iter + 1):
        gradient = (codes @ dictionary - x) @ dictionary.T + penalty
        codes = (codes - gradient / lipschitz).clamp_min(0)
        gradient = (codes @ dictionary - x) @ dictionary.T + penalty
        # Fixed-point residual scaled back into gradient units. Zero iff KKT.
        residual = lipschitz * (codes - (codes - gradient / lipschitz).clamp_min(0))
        kkt = residual.abs().amax(1)
        converged = kkt <= threshold
        if bool(converged.all()):
            break
    reconstruction = codes @ dictionary
    if not codes.isfinite().all() or not reconstruction.isfinite().all() or not kkt.isfinite().all():
        raise ValueError("Sparse optimization overflowed; rescale inputs or use float64")
    error = x - reconstruction
    error_norm = error.norm(dim=1)
    x_norm = x.norm(dim=1)
    rec_norm = reconstruction.norm(dim=1)
    cosine_defined = (x_norm > eps) & (rec_norm > eps)
    denominator = torch.where(cosine_defined, x_norm * rec_norm, torch.ones_like(x_norm))
    cosine = (x * reconstruction).sum(1) / denominator
    cosine = torch.where(cosine_defined, cosine, torch.full_like(cosine, float("nan")))
    raw_reconstruction = reconstruction * target_norm + target_mean if normalize else reconstruction + target_mean
    objective = 0.5 * error.square().sum(1) + penalty * codes.sum(1)
    if not objective.isfinite().all() or not raw_reconstruction.isfinite().all():
        raise ValueError("Reconstruction diagnostics overflowed; rescale inputs or use float64")
    return {"codes": codes, "dictionary": dictionary, "target": x,
            "reconstruction": reconstruction, "residual": error,
            "raw_reconstruction": raw_reconstruction,
            "objective": objective,
            "reconstruction_error": error_norm,
            "relative_reconstruction_error": error_norm / x_norm.clamp_min(eps),
            "cosine_similarity": cosine, "cosine_defined": cosine_defined,
            "active_concepts": (codes > 0).sum(1),
            "kkt_residual": kkt, "convergence_threshold": threshold,
            "converged": converged, "iterations": iteration,
            "lipschitz": lipschitz, "penalty": penalty,
            "embedding_mean": target_mean, "vocabulary_mean": concept_mean,
            "target_norm": target_norm, "vocabulary_norm": dictionary_norm,
            "normalized": normalize, "method": "splice_style_nonnegative_l1_ista"}


def reconstruct_concepts(codes, dictionary, *, mean=None, row_scale=None):
    """Reconstruct in dictionary coordinates; optionally undo target preprocessing.

    dictionary must be the returned transformed dictionary, not raw vocabulary.
    row_scale=result['target_norm'] and mean=result['embedding_mean'] undo the
    normalized target transform. If normalization was disabled, omit row_scale.
    Outputs are embedding reconstructions, never decoded images/text.
    """
    codes, dictionary = _matrix(codes, "codes"), _matrix(dictionary, "dictionary")
    _compatible(codes, dictionary)
    if codes.shape[1] != dictionary.shape[0] or (codes < 0).any():
        raise ValueError("Nonnegative codes must match the dictionary concept count")
    reconstruction = codes @ dictionary
    if row_scale is not None:
        row_scale = torch.as_tensor(row_scale, device=codes.device, dtype=codes.dtype)
        if (row_scale.shape != (codes.shape[0], 1) or not row_scale.isfinite().all()
                or (row_scale < 0).any()):
            raise ValueError("row_scale must be finite nonnegative [batch,1]")
        reconstruction = reconstruction * row_scale
    if mean is not None:
        _, mean = _center(reconstruction, mean, "mean")
        reconstruction = reconstruction + mean
    if not reconstruction.isfinite().all():
        raise ValueError("Reconstruction overflowed")
    return reconstruction


def intervene_concepts(target, codes, dictionary, *, concept, value=None, scale=None,
                       preserve_error=True):
    """Edit one concept in the transformed target space; return embedding and codes.

    Use result['target'], ['codes'], ['dictionary'] from the solver. Exactly one
    nonnegative finite `value` (replacement) or `scale` is required. By default
    target + (new_codes-old_codes) @ dictionary retains the reconstruction error
    and is exactly identity for a zero coefficient change. The output is NOT
    renormalized, fed into an encoder, or interpreted as a causal input edit.
    Apply the saved target scale/mean separately to recover raw coordinates.
    """
    target = _matrix(target, "target")
    codes, dictionary = _matrix(codes, "codes"), _matrix(dictionary, "dictionary")
    _compatible(target, codes)
    original = reconstruct_concepts(codes, dictionary)
    if target.shape != original.shape:
        raise ValueError("Target must match the reconstructed batch and embedding width")
    if type(concept) is not int or not 0 <= concept < codes.shape[1]:
        raise ValueError("concept must index the vocabulary")
    if (value is None) == (scale is None):
        raise ValueError("Supply exactly one replacement value or multiplicative scale")
    magnitude = value if value is not None else scale
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("Intervention magnitude must be finite and nonnegative")
    changed = codes.clone()
    changed[:, concept] = value if value is not None else codes[:, concept] * scale
    # Compute the delta directly to preserve the identity edit without roundoff
    # from subtracting two independently reconstructed embeddings.
    delta = (changed - codes) @ dictionary
    edited = target + delta if preserve_error else reconstruct_concepts(changed, dictionary)
    if not edited.isfinite().all():
        raise ValueError("Intervention overflowed")
    return {"embedding": edited, "codes": changed, "delta": delta,
            "preserve_error": preserve_error}


@torch.no_grad()
def modality_feature_diagnostics(image_features, text_features, *, threshold=0.0, paired=False):
    """Descriptive shared/modality-specific statistics for aligned feature axes.

    Nonnegative [Nimage,K], [Ntext,K] feature activations must use the SAME
    vocabulary or aligned SAE dictionary. Positive specificity favors images;
    negative favors text. Shared prevalence=min(image prevalence,text prevalence)
    does not imply semantic equivalence or causal sharing. `paired=True` adds
    per-feature paired correlation and coactivation, requiring matched rows.
    No feature alignment, SAE fitting, hierarchy, or multimodal loss is learned.
    """
    image = _matrix(image_features, "image_features")
    text = _matrix(text_features, "text_features")
    _compatible(image, text)
    if image.shape[1] != text.shape[1] or (image < 0).any() or (text < 0).any():
        raise ValueError("Need nonnegative features with an explicitly aligned feature width")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    image_active, text_active = image > threshold, text > threshold
    image_prevalence = image_active.to(image.dtype).mean(0)
    text_prevalence = text_active.to(text.dtype).mean(0)
    image_mean, text_mean = image.mean(0), text.mean(0)
    eps = torch.finfo(image.dtype).eps
    result = {"image_prevalence": image_prevalence, "text_prevalence": text_prevalence,
              "image_mean": image_mean, "text_mean": text_mean,
              "shared_prevalence": torch.minimum(image_prevalence, text_prevalence),
              "prevalence_difference": image_prevalence - text_prevalence,
              "mean_specificity": (image_mean - text_mean) / (image_mean + text_mean).clamp_min(eps),
              "image_count": image.shape[0], "text_count": text.shape[0]}
    if paired:
        if image.shape != text.shape or image.shape[0] < 2:
            raise ValueError("Paired diagnostics need at least two matched observations")
        a, b = image - image_mean, text - text_mean
        denominator = a.norm(dim=0) * b.norm(dim=0)
        defined = denominator > eps
        correlation = (a * b).sum(0) / denominator.clamp_min(eps)
        result.update({"paired_correlation": torch.where(defined, correlation.clamp(-1, 1),
                                                        torch.full_like(correlation, float("nan"))),
                       "paired_correlation_defined": defined,
                       "paired_coactivation": (image_active & text_active).to(image.dtype).mean(0)})
    return result
