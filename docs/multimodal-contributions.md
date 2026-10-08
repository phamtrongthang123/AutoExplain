# Structured multimodal contributions

`autoexplain.multimodal` supplies **forward-only mathematical primitives**, not
verified end-to-end reproductions of MM-SHAP, InterSHAP or explanation-reliance
experiments. Whole named modalities are the players. No model downloads,
additional dependencies, architecture inference or automatic target selection
are involved.

## Score and reference contract

```python
from autoexplain.multimodal import MultimodalScorer, modality_shapley

# image and tokens have the same leading batch dimension and device.
# Keep a fixed target class under every intervention.
scorer = MultimodalScorer(
    lambda parts: model(parts["image"], parts["tokens"])[:, target_class],
    model=model,
)
result = modality_shapley(
    scorer,
    {"image": image, "tokens": tokens},
    {"image": reference_image, "tokens": reference_tokens},
)
# result.names identifies columns of result.contributions [B, M].
```

Inputs are a nonempty mapping from nonempty string names to nonempty, finite,
real, dense PyTorch tensors `[B, ...]`. Integer token IDs and boolean masks are
allowed. All modalities share the same batch size and device; trailing shapes
and dtypes can differ. Mapping insertion order determines modality indices.
A modality with multiple dependent tensors (tokens plus mask, for example) must
be represented by an explicit caller packing convention or reconstructed by the
score adapter. Listing them as separate names makes them separate players; the
library does not infer which components must change together.

The callable receives a new dictionary of cloned tensors and must return a
finite floating tensor **exactly `[B]` on the input device**. Reduce class logits,
sequence log likelihoods or spatial outputs explicitly. Objectives, targets,
reference tokens and explanation strings must remain fixed under intervention;
reselecting the predicted class or regenerating the explanation changes the
question being answered. A fixed per-row target is possible for coalition games,
but EMAP changes row identity and batch size: use a shared objective or carry
needed conditioning inside the appropriate modality and adapt it explicitly.

Every reference has exactly the corresponding input dtype/device and either
the full input shape or `[1, ...]` with identical trailing shape. Only the
leading singleton dimension is broadcast. References are **deterministic
replacement tensors**, not samples integrated over a background distribution,
conditional expectations, missing keys, or inferred semantic absence. All
modality names must have a reference and no extra reference keys are accepted.
Use meaningful references: zeros or pad tokens can be out of distribution.

`MultimodalScorer(score_fn, *, model=None)` also supports direct calls on named
inputs. An `nn.Module` score callable is temporarily evaluated; a closure around
a model must pass `model=model` so its training flags can be managed. All managed
submodule training flags are restored, including after exceptions. Calls use
`torch.no_grad()` and do not clear or accumulate parameter gradients. Returned
scores are detached float64 on the input device, requiring float64 support
(e.g. CPU or CUDA). Caller input storage is protected against adapter in-place
writes. Arbitrary callable state, hidden unregistered models, random number
state and model buffer mutation cannot be restored by this API. Avoid concurrent
training/evaluation of the same module during these calls.

Scores must be deterministic and independent across batch rows. Stateful
scorers, batch-dependent objectives, training-mode batch normalization, cross-row
attention and fresh random sampling break the interpretation. The library
validates shapes/devices/finiteness, not these behavioral assumptions. Arithmetic
overflow raises an error instead of silently returning infinite diagnostics.

## Exact modality Shapley and pair interactions

`modality_shapley(scorer, inputs, references, *, max_coalitions=256,
max_evaluations=1_000_000)` enumerates all `2**M` coalitions. Define `v(S)` as the
score with modalities in `S` from the observation and all others from their
references, separately for each sample. Then

\[
\phi_i=\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(M-|S|-1)!}{M!}\,[v(S\cup\{i\})-v(S)].
\]

The symmetric interaction matrix uses the **SHAP half-interaction convention**:

\[
I_{ij}=\sum_{S\subseteq N\setminus\{i,j\}}
\frac{|S|!(M-|S|-2)!}{2(M-1)!}
[v(Sij)-v(Si)-v(Sj)+v(S)],\quad i\ne j,
\]

and `I[i,i] = phi[i] - sum(j != i, I[i,j])`. Thus each row sums to
`phi[i]`, and the whole matrix sums to `v(N)-v(empty)` up to floating-point
error. An unordered pair's total contribution is `2*I[i,j]`, not `I[i,j]`.
Diagonals are remaining main effects under this convention, not raw singleton
ablations. With one modality there are no pairs and the diagonal is its Shapley
value. With two modalities each off-diagonal is half the four-corner second
difference. Signed contributions and interactions can be negative.

The result includes `names`, `coalition_scores [B, 2**M]` (integer bitmasks in
column order), `contributions [B,M]`, `interactions [B,M,M]`, and
`efficiency_residual [B] = sum(phi) - (full-reference)`.

The default coalition budget allows at most eight modalities, and may be
explicitly raised. `max_evaluations` counts **scored rows**, not forward calls;
`B*2**M` must fit before the first forward. Cost is `2**M` forward calls of
batch size `B`, `O(B*M**2*2**M)` arithmetic for interactions and
`O(B*(2**M+M**2))` score/result storage, excluding inputs/model activations.
This is exact for the specified replacement game, not an estimate of an
unknown natural missing-modality distribution.

## Ablations and mismatched pairs

`modality_ablations(scorer, inputs, references, *, max_evaluations=1_000_000)`
returns full/reference scores `[B]`, `without_scores` and `only_scores [B,M]`,
`removal_effects = full-without`, and `addition_effects = only-reference`.
It evaluates the distinct coalitions among empty/full, each leave-one-out and
each singleton: at most `2*M+2` calls and `(2*M+2)*B` rows. These effects need
not sum to the total and are not Shapley attributions.

`mismatched_pairs(scorer, inputs, *, modality, permutation=None,
max_evaluations=1_000_000)` compares original inputs with one modality permuted
across observations. It needs `B >= 2`, two forward calls and `2*B` scored rows.
Default permutation is a deterministic one-position cyclic shift, not a random
null distribution. An explicit integer `[B]` tensor on the input device must
be a bijection with no fixed points; changed row `i` receives original row
`permutation[i]`. The result records the permutation, matched/mismatched scores
and their signed difference. Distinct row indices can still contain identical
content. Choose semantically suitable independent observations and account for
labels/conditioning retained in the score adapter. No uncertainty estimates,
causal effects or statistical significance are inferred.

## Two-modality EMAP

`emap(scorer, inputs, *, batch_size=128, max_pairs=65_536)` requires exactly two
modalities and `N` paired observations. It **explicitly evaluates all `N*N`
Cartesian pairs**, in row-major order and chunks no larger than `batch_size`:

\[
F_{ij}=f(x_i,y_j),\quad \mu=N^{-2}\sum_{ij} F_{ij},\qquad
\widehat F_{ij}=\overline F_{i\cdot}+\overline F_{\cdot j}-\mu.
\]

This is the least-squares additive projection under the empirical **product of
marginals**, not a regression on only the observed diagonal and not a projection
under the original dependent joint distribution. Centered modality effects are
`row_mean-mu` and `column_mean-mu`. The residual `F - projected` has zero row and
column means up to rounding. No replacement references are used. `N=1` is
allowed and yields a trivial zero-residual projection.

`EMAPResult` exposes `names`, `pair_scores [N,N]`, scalar `grand_mean`, centered
`first_effects`/`second_effects [N]`, `additive_scores`/`residuals [N,N]`, and the
observed diagonal as `matched_scores`/`matched_additive_scores [N]`. Chunking
bounds forward-batch size but does **not** remove the quadratic score storage.
The budget rejects `N*N > max_pairs` before scoring. There are
`ceil(N*N/batch_size)` forward calls, `N*N` scored rows and `O(N*N)` storage.
There is no sampled approximation or out-of-sample extension. For vector-output
EMAP, invoke this scalar-objective implementation for each required coordinate;
it does not return multiclass predictions or evaluate downstream accuracy.

## Answer/explanation reliance comparison

`compare_answer_explanation_reliance(answer_scorer, explanation_scorer, inputs,
references, *, max_coalitions=256, max_evaluations=1_000_000)` applies the same
fixed-reference game to two caller-supplied fixed scalar objectives. For example,
one might score a fixed answer and the other the teacher-forced likelihood of a
fixed explanation, with length normalization explicitly chosen by the caller.

It returns both complete Shapley results, each normalized absolute contribution
vector `abs(phi)/sum(abs(phi))`, and per-row L1 distance in `[0,2]`. A row is
`valid` only when both absolute totals are positive. Undefined share vectors
are represented by zeros and their comparison distance by **NaN**, not a claim
of perfect agreement. There is no near-zero threshold; inspect raw magnitudes
for numerical instability. Absolute shares discard contribution signs and
score scale. The budget counts rows **combined across both games**; forward
work is twice that of one Shapley call. Similar shares alone do not show that an
explanation is truthful, causally faithful, useful or textually consistent.

## Provenance and deliberately unsupported parts

These implementations follow the equations stated above. The following papers
motivate related analyses; their complete model/data/benchmark pipelines have
**not** been implemented or validated by this module:

- [MM-SHAP, ACL 2023](https://aclanthology.org/2023.acl-long.223/): this module
  provides whole-modality exact replacement Shapley and normalized absolute
  reliance primitives. It does not reproduce token/patch feature coalitions,
  paper-specific masking, feature-to-modality aggregation, datasets or reported
  MM-SHAP metrics. Whole-modality grouping generally changes the game and is not
  interchangeable with summing feature-level Shapley values.
- [InterSHAP, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35452):
  the pair matrix here uses the explicit SHAP interaction convention above.
  No claim is made that this alone reproduces InterSHAP's full procedure,
  interaction summaries, background sampling or evaluation protocol.
- [EMAP, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.62/): the exact
  two-modality empirical Cartesian additive projection is implemented for one
  scalar output. Paper-specific classifiers, training, datasets, vector-output
  prediction decisions and benchmark comparisons are not included.
- [Cross-modal influence, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.775/):
  ablations and mismatched pairs are generic diagnostic controls, not a
  reproduction of that paper's influence estimator or analysis pipeline.
- [Decoder explanation reliance](https://arxiv.org/abs/2404.18624): the comparison
  here needs explicit fixed answer/explanation score adapters. It does not
  implement generation, paper-specific prompts, masking, sequence scoring or
  evaluation benchmarks.

No empirical reproduction, accuracy or faithfulness claim follows from these
primitives. This implementation batch was statically reviewed; no tests or model
experiments were run.
