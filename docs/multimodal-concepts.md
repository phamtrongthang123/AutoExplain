# CLIP attribution and sparse multimodal concepts

These modules operate on **caller-provided encoders, internal tensors and
vocabulary embeddings**. They download nothing and add no dependencies. Import
from `autoexplain.clip_tools` or `autoexplain.multimodal_concepts` directly.

## Support boundaries

| Workflow | Implemented | Not claimed |
| --- | --- | --- |
| CLIP score targets | Differentiable normalized image/text dot products, explicit aligned-pair or indexed text targets, optional positive logit multiplier | Tokenization, preprocessing, pretrained model loading, implicit softmax targets |
| Grad-ECLIP | Equations (7)–(8) with official-code cosine/min–max spatial weights, from explicit single-head forward intermediates | Full encoder instrumentation, unchanged native multi-head CLIP equivalence, benchmark reproduction |
| Token attribution | Signed gradient-times-token, explicit rectangular patch layouts | Equivalence to Grad-ECLIP, attention rollout, causal attribution |
| Head/patch/text decomposition | Exact linear attention-value bookkeeping and additive projected-space score accounting | Complete ICLR 2024 text-based CLIP decomposition through arbitrary LayerNorm/MLP blocks |
| SpLiCE-style concepts | Nonnegative L1 sparse coding with an explicit dictionary, centering and normalization, reconstruction and optimization diagnostics | Exact SpLiCE reproduction, vocabulary release, authors' solver or calibration pipeline |
| Concept interventions | Error-preserving coefficient edits in the declared embedding space | Image/text generation, an input-space causal effect |
| Cross-modal feature diagnostics | Prevalence, activation specificity, paired correlation/coactivation on aligned axes | Trained hierarchical SAE, SAE-V, VL-SAE or a complete concept-based LMM pipeline |

Grad-ECLIP is implemented as an **equation-grounded, single-layer primitive**,
not a drop-in adapter for arbitrary CLIP models. Direct reading of the primary
paper and linked official implementation established the contract below after
the initial public-search timeout. Generic token gradients remain a separately
named primitive, never a Grad-ECLIP substitute.

### Grad-ECLIP: exact intermediate contract

The paper's Eqs. (7)–(8) define

`H[i] = ReLU(sum_c w_channel[c] * w_spatial[i] * v[i,c])`,

where `w_channel = d(cosine_matching_score) / d(attention_output[CLS/EOS])`.
The authors' `generate_emap.py` supplies the concrete spatial normalization:

`r[i] = cosine(q_out[CLS/EOS], k_out[i])`,

`w_spatial[i] = (r[i] - min(r)) / (max(r) - min(r))`.

This is a loosened attention weight, **not softmax attention or rollout**.
Channel weights are the selected attention-output row's gradients, not averaged
patch-token gradients as in Grad-CAM. The official implementation applies the
attention output projection **including its bias** to raw Q and K to produce
`q_out` and `k_out`, but uses **raw V before that output projection** in the
weighted sum. Do not replace raw V by `W_O V` or take the gradient after `W_O`.

```python
from autoexplain.clip_tools import grad_eclip, patch_attribution_map

# Caller instruments the forward; these are NOT obtainable from final
# image/text embeddings alone. attention_output must already be a [B,Q,C]
# ancestor used in computing scores, not a newly permuted post-hoc view.
result = grad_eclip(
    scores, attention_output,
    values=v[:, 1:, :],             # image patches, exclude CLS
    query=q_out[:, 0, :],
    keys=k_out[:, 1:, :],
    query_index=0,                  # CLS row of attention_output
    attention_mode="single_head",  # explicit caller assertion
)
heatmap = patch_attribution_map(result['relevance'], grid_shape=(rows, columns))
```

Required shapes are `scores [B]`, `attention_output [B,Q,C]`, `values [B,T,C]`,
`query [B,D]` and `keys [B,T,D]`. The selected values and keys must use the same
token ordering. `query_index` may be a scalar or one integer per batch row.
Scores must be unscaled cosine matching scores from the same forward. Batch
rows must be independent; gradients are computed with a summed-score VJP and
do not accumulate into parameter `.grad` buffers.

**Single-head forward is a material assumption.** Paper p. 4 and the official
`clip_encode_dense` / `clip_encode_text_dense` explicitly form attention with
one head across the full channel width, instead of the original per-head
softmax. The score and intermediates must come from that consistent forward,
including the original subsequent output projection, residual/MLP and final
embedding operations. The helper requires `attention_mode="single_head"` and
rejects other declarations, but cannot inspect or certify a caller's model.
Merely concatenating native multi-head activations does not meet this contract.
The repository does not silently rewrite the user's model, so full encoder
instrumentation remains the caller's responsibility. Explaining this modified
forward must not be described as an exact explanation of an unchanged native
multi-head CLIP score.

The returned `relevance` is the single-layer ReLU map; `signed_relevance`
retains pre-ReLU contributions. Channel/spatial weights and query/key cosine
are also returned. Constant or numerically constant cosine maps have undefined
min–max normalization: our deliberate numerical extension returns zero weights
and maps with `spatial_weights_defined=False`, rather than the official code's
division by zero. Zero/degenerate query or key vectors are rejected. Half and
bfloat16 calculations are promoted to float32, and there is no hidden map
normalization, image resizing or interpolation.

For text, select the EOS attention-output row and its projected query, but
include **all sequence keys and values** when computing each layer's spatial
normalization to match the official code (including special/padding tokens).
The paper uses the last eight text layers; for a supplied set of correctly
instrumented layers, call this primitive on each with `retain_graph=True`
until the last call, sum **signed_relevance** across layers, then apply ReLU.
Only afterward crop BOS/EOS/padding and divide by the retained relevance sum
if positive. Applying ReLU to each layer before summing is not the authors'
text aggregation. Empty/zero retained text maps must not be divided by zero.
No tokenizer, EOS discovery or multi-layer encoder adapter is supplied.

## Explicit differentiable scores

```python
from autoexplain.clip_tools import encode_clip_scores, clip_similarity

# Encoders must return projected [batch, embedding_width] tensors.
# Caller handles eval mode, tokenizer/masks, and image normalization.
scores = encode_clip_scores(
    images, token_ids,
    image_encoder=model.encode_image,
    text_encoder=model.encode_text,
    target=2,                         # same text index for every image
    scale=model.logit_scale.exp(),    # multiplier, NOT the stored log scale
)
# For unscaled cosine, omit scale. target=None selects aligned pairs and
# requires equal image/text batch sizes. An integer vector selects per-image.
all_cosines = clip_similarity(image_embeddings, text_embeddings)
```

Scores are not probabilities. `clip_similarity` returns `[images,texts]`, while
`clip_scores` and `encode_clip_scores` return one selected score per image.
Normalization remains in the autograd graph. Zero/degenerate embeddings raise
instead of producing misleading cosine values. Half/bfloat16 are promoted to
float32; otherwise promoted dtypes and devices must agree. The helpers do not
switch model modes, freeze encoders, accumulate parameter gradients or detach
scores.

### Image patch or text token gradients

`token_gradient_attribution(scores, tokens)` requires the actual floating
`[B,T,C]` tensor consumed in the scoring graph. Integer text token IDs are not
differentiable: capture the embedding/block tensor instead. Do not construct a
new post-hoc view and expect it to be an ancestor of already-computed scores.
The routine uses `autograd.grad(scores.sum(), tokens)` and returns signed
`(gradient * token).sum(-1)` and the gradients, without changing `.grad` buffers.
The batch must be independent: batch-coupled operations invalidate the
per-example interpretation of this summed-score derivative. The graph is freed
unless `retain_graph=True` is requested.

`patch_attribution_map(relevance, grid_shape=(rows, columns),
patch_indices=indices)` explicitly orders patch tokens into a rectangular grid.
It never guesses a square, removes a CLS/register token implicitly, or reverses
cropping/resize operations. Indices must be distinct and match the grid area.

### Exact but bounded head/token bookkeeping

For one selected attention query:

- `attention_value_decomposition(attention, values, output_weight,
  output_bias=...)` accepts weights `[B,H,T]`, projected values `[B,H,T,Dh]`,
  and the `torch.nn.Linear` output projection weight `[Dout,H*Dh]`.
- It computes each term `a[h,t] * W_O[:,h] @ v[h,t]`, returning
  `[B,H,T,Dout]` contributions, explicit bias, and their summed output.
- Weights must be the ones actually used in the forward (including attention
  dropout if enabled). Query/key effects are not separated; these terms are
  descriptive, not interventions. Grouped-query attention must first be mapped
  to the actual value/head layout by the caller.

`projected_component_scores(components, reference, remainder=...)` works only
when `[B,K,D]` components and the remainder sum to the **full pre-normalized
embedding in the shared projected space**. It reports

`contribution[k] = scale * dot(component[k], unit(reference)) / norm(full)`.

The common denominator is essential: separately normalizing components does
not yield additive cosine scores. The returned remainder score, direct score,
and additivity residual make omissions visible. Bias/residual pathways belong
in the remainder. Passing raw intermediate block heads without transporting
them through all later transformations is not an exact CLIP decomposition.

## SpLiCE-style sparse nonnegative dictionary coding

```python
from autoexplain.multimodal_concepts import (
    sparse_concept_decomposition, reconstruct_concepts, intervene_concepts,
    modality_feature_diagnostics,
)

result = sparse_concept_decomposition(
    image_embeddings, vocabulary_embeddings,
    embedding_mean=image_calibration_mean,
    vocabulary_mean=text_calibration_mean,
    penalty=0.01, normalize=True, max_iter=2000, tolerance=1e-5,
)
# Vocabulary row i has the caller's label i; no label is inferred by this solver.
# Inspect result['converged'] and result['kkt_residual'] before interpreting codes.
raw_approximation = reconstruct_concepts(
    result['codes'], result['dictionary'],
    row_scale=result['target_norm'], mean=result['embedding_mean'],
)
edit = intervene_concepts(
    result['target'], result['codes'], result['dictionary'],
    concept=7, scale=0.0, preserve_error=True,
)
raw_edited_embedding = edit['embedding'] * result['target_norm'] + result['embedding_mean']
```

Let `X` be the transformed target matrix and `D` the transformed vocabulary
matrix, one concept per row. The objective is explicitly

`min_(Z >= 0) 0.5 * ||X - Z D||_F^2 + penalty * sum(Z)`.

The implementation uses proximal gradient (ISTA):

`Z_next = max(0, Z - ((Z D - X) D.T + penalty) / L)`,

with `L = ||D||_2^2` computed once by a spectral norm. This is an L1-penalized
**SpLiCE-style variant**, not a claim of exact paper reproduction. It does not
implement a separate constrained sparsity budget or automatically match the
paper's hyperparameters. The spectral-norm calculation and dense codes can be
expensive for a large vocabulary; there is no automatic sampling or hidden
GPU job.

### Coordinate and calibration assumptions

1. Input and vocabulary embeddings must be in a compatible shared latent space.
2. Supplied modality means are subtracted independently. `None` means zero,
   **not** an implicitly estimated mean. Estimate means on disjoint calibration
   data and document whether separate modality centering is appropriate.
3. With `normalize=True`, each centered target and dictionary row is normalized
   to unit norm. Zero centered rows are rejected; explicitly filter degenerate
   vocabulary entries and retain your row-to-label mapping.
4. With `normalize=False`, zero targets are valid and produce zero codes;
   dictionary rows must still be nondegenerate. Unequal dictionary norms then
   change the effective L1 preference. Raw reconstruction only adds the target
   mean; do not multiply by `target_norm` in that case.
5. Coefficients are not probabilities. Correlated/duplicate vocabulary items
   can make their allocation nonunique even when reconstruction is stable.
   Sparsity and a good cosine fit do not establish a concept's causal validity.

Returned diagnostics include per-row objective, reconstruction and relative
error, reconstruction cosine, number of strictly positive coefficients,
projected-gradient/KKT residual, convergence threshold and flag, plus iteration
count and Lipschitz constant. The stopping residual is
`L * ||Z - max(0, Z - gradient/L)||_infinity` per row, with threshold
`tolerance * max(1, ||initial_gradient||_infinity)`. Exhausting `max_iter` does
not become success. A zero reconstruction has undefined cosine: it is returned
as NaN with a separate `cosine_defined` mask, not misleadingly reported as zero
or one. The solver is detached analysis code, not differentiable training.

### Reconstruction-preserving interventions

`intervene_concepts` edits one nonnegative coefficient by replacement or scale.
With `preserve_error=True` it returns

`target + (edited_codes - original_codes) @ dictionary`.

The original residual therefore remains present. A zero coefficient change is
identity without subtracting separately rounded reconstructions. With
`preserve_error=False`, it returns the edited dictionary reconstruction instead.
The output stays in transformed coordinates and is not unit-normalized. To
measure a change in CLIP similarity, undo the declared transform if appropriate,
then explicitly score/normalize again. No embedding-to-image decoder is implied.

## Shared versus modality-specific feature diagnostics

`modality_feature_diagnostics(image_features, text_features, threshold=0,
paired=False)` accepts nonnegative matrices with the **same aligned feature
axes** (a common vocabulary or an already aligned SAE dictionary). It reports
activation prevalence, mean activation, shared prevalence (the minimum of the
two prevalences), prevalence difference and signed mean specificity. Positive
specificity favors images; negative favors text. All-zero features have zero
specificity. Unpaired observation counts may differ.

For matched image/text rows, `paired=True` adds coactivation and Pearson
correlation. Constant axes have NaN correlations plus a validity mask. These
are descriptive population statistics, not significance tests. Activation
scales, calibration set composition, threshold choice and correspondence all
matter. Comparing independently trained SAE axes without alignment is invalid.

There is no SAE training here: hierarchical feature assignment, multimodal
reconstruction/alignment losses, dead-feature handling, architecture-specific
hooks, checkpoint conversion and paper-specific training/evaluation recipes
remain necessary for hierarchical SAE, SAE-V and VL-SAE reproductions. Generic
coefficient diagnostics do not substitute for those methods. Likewise, concept
coding alone does not implement an LMM's concept-to-output causal pipeline.

## Source references and verification status

- Grad-ECLIP primary page: <https://proceedings.mlr.press/v235/zhao24p.html>
  - Directly read paper pp. 4–5, Eqs. (5)–(8):
    <https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhao24p/zhao24p.pdf>
  - Directly read the linked official implementation (`grad_eclip`,
    `grad_eclip_text`, `clip_encode_dense`, `clip_encode_text_dense`):
    <https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/generate_emap.py>
    (retrieved Git blob SHA `3319278ff05edf6c4278383594ab9cfecbd7ac86`).
- CLIP text-based decomposition (ICLR 2024):
  <https://proceedings.iclr.cc/paper_files/paper/2024/hash/5085a2b8c298edeadc46b9ffe6df5f64-Abstract-Conference.html>
- SpLiCE: <https://arxiv.org/abs/2402.10376>
- Concept-based LMM framework: <https://arxiv.org/abs/2406.08074>
- Hierarchical SAE: <https://proceedings.mlr.press/v267/zaigrajew25a.html>
- SAE-V: <https://proceedings.mlr.press/v267/lou25b.html>
- VL-SAE: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html>

Grad-ECLIP's equations and intermediate contract were checked against the
primary paper and official code above; other references identify the associated
papers, not verified full reproductions. Implementation received static
code/diff review only; no tests, builds, model downloads or training were run.
