# Explicit multimodal evaluation and inspection protocols

These standalone PyTorch utilities implement grouped evidence interventions,
annotation overlap, fixed-answer causal-LM scoring, clean/corrupt activation
patching, and serializable attention inspection. They require explicit modality
layouts, replacements, scoring choices and alignment. They do **not** infer these
from a model name, implement complete paper pipelines, or equate visual
plausibility with model faithfulness.

Import directly from `autoexplain.multimodal_metrics` and
`autoexplain.multimodal_tracing`. No optional backend or new dependency is needed.
Model execution remains the caller's responsibility; examples below illustrate
contracts, not validated support for a particular pretrained architecture.

## Grouped deletion, insertion, sufficiency and necessity

`EvidenceGroup(name, modality, mask)` describes a boolean mask over a named
input's **non-batch** dimensions. For example:

- `[T]` for token IDs `[B,T]`;
- `[T,1]` for token embeddings `[B,T,D]`;
- `[1,H,W]` for image inputs `[B,C,H,W]`;
- `[F,1,1,1]` for video `[B,F,C,H,W]`.

Irregular regions, rectangular patch layouts and noncontiguous frame sets work:
there is no square-grid assumption. Groups must be nonempty, uniquely named and
disjoint within a modality after broadcasting. The same masks and ranking apply
to every example in the batch; use separate calls for per-example layouts or
rankings. A group can cover one modality, not several; use multiple named groups
for multi-modality evidence. Group masks select scalars after broadcasting, so a
region broadcast over channels has its area multiplied by the channel count.

```python
from autoexplain.multimodal_metrics import EvidenceGroup, grouped_evidence_evaluation

# model(**inputs) returns an explicitly adapted [B, classes] tensor.
# target_ids is fixed before evaluation and has shape [B].
def chosen_class(output):
    return output.gather(1, target_ids.to(output.device)[:, None]).squeeze(1)

result = grouped_evidence_evaluation(
    model,
    {"image": images, "question_ids": question_ids},
    groups=[
        EvidenceGroup("object", "image", object_mask),
        EvidenceGroup("context", "image", context_mask),
        EvidenceGroup("question_phrase", "question_ids", phrase_mask),
    ],
    replacements={"image": replacement_images, "question_ids": replacement_ids},
    importance=[0.8, 0.2, 0.5],
    score_fn=chosen_class,
    budgets=[0, 1, 2, 3],
    random_trials=5,
    generator=torch.Generator().manual_seed(7),
)
```

The masks in this example must be disjoint within each modality. Replacement
values are full-shape tensors, exactly matching each intervened input's device
and dtype. Other named inputs (including attention masks) are passed unchanged;
if a replacement requires a changed padding mask or model-specific preprocessing,
provide a coherent adapter instead of assuming the evaluator updates it. All
named inputs must be real, dense, nonempty batched tensors with the same batch
size. Every input—including nonintervened inputs—and replacement must be finite.
Integer token IDs and boolean tensors remain supported; complex, sparse,
quantized, nested and meta tensors are rejected. Group entries must be
`EvidenceGroup` instances, not dictionaries or arbitrary objects.

Importance is signed and sorted descending, with stable tie handling. Explicitly
pass absolute importance if that is the intended ranking. Each budget counts
**groups**, not scalar features, and the returned fractions are budget / total
groups. Elements outside all groups always remain original. Thus full insertion
means all grouped evidence restored, and insertion at budget zero is only a
replacement of the grouped universe—not necessarily a blank modality.

Returned CPU float64 fields include:

| Field | Meaning / shape |
| --- | --- |
| `original_scores`, `replacement_scores` | Original and grouped-universe replacement baselines, `[B]` |
| `deletion_scores` | Replace the first k ranked groups, `[B,K]` |
| `insertion_scores` | Keep only the first k ranked groups within the grouped universe, `[B,K]` |
| `sufficiency_gap` | Original minus insertion; lower may indicate sufficient retained evidence |
| `necessity_drop` | Original minus deletion; larger may indicate necessary removed evidence |
| `deletion_auc`, `insertion_auc` | Trapezoidal AUC over supplied group fractions, `[B]` |
| `random_*_scores`, `random_sufficiency_gap`, `random_necessity_drop` | Matched controls, `[R,B,K]` |
| `random_deletion_auc`, `random_insertion_auc` | Control AUCs, `[R,B]` |
| `group_order`, `random_orders`, `budgets`, `fractions` | Reproducible group identity and budget metadata |

Random permutations are stratified by **modality and expanded scalar group
size**. At every prefix budget they match group count, modality composition and
scalar replacement count. They do not match content, geometry or semantics;
singleton strata cannot vary. Supply a PyTorch generator (CPU or its own device)
for explicit RNG ownership. Without one, the CPU global RNG is used. Defaults cap
groups at 2,048 and requested model forwards at 10,000; larger sweeps must opt in.
AUC over a partial budget range is not a full-range AUC. A single budget gives
zero trapezoidal area and should not be interpreted as a curve comparison.

Scores must be finite floating `[B]` values for the **same chosen output** on
every forward. A score callback that reselects argmax would change the question
being evaluated and violates this contract. No automatic label choice or
probability interpretation is made. Masking may introduce out-of-distribution
inputs: report replacement policy and controls, not just one AUC. Sufficiency and
necessity are properties of this chosen intervention, score and grouped universe.

## Grounded evidence overlap is not faithfulness

`grounded_evidence_metrics(predicted, reference, semantics=..., weights=None)`
accepts same-shaped boolean `[B,...]` masks in an explicitly shared coordinate
system. It returns per-sample weighted precision, recall, IoU, Dice, intersection,
prediction/reference mass and empty flags. `semantics` is one of `spatial`,
`temporal`, `spatiotemporal`; this is explicit caller metadata, not an inferred
axis layout. The function does not threshold saliency, rasterize boxes, resample
annotations, or choose an IoU matching policy for separate objects.

Temporal bins represent half-open `[edge_i, edge_{i+1})` intervals. For unequal
bins, pass duration weights; spatial weights can encode physical pixel/region
area, and joint masks may use space-time volume. Nonnegative finite weights must
broadcast to `[B,...]`. A `[T]` temporal weight vector works for `[B,T]`; for
`[B,T,H,W]`, reshape it to `[1,T,1,1]`. Uniform weights count equal-sized bins.
Zero-weight support contributes nothing and is treated as empty.

Empty policies are deliberate:

- both masks empty: precision, recall, IoU and Dice are 1;
- predicted empty only: precision 1, recall 0, IoU/Dice 0;
- reference empty only: precision 0, recall 1, IoU/Dice 0.

Report empty flags or stratified results so annotation-free examples do not
inflate an aggregate. No aggregate across examples is silently computed. An
annotation overlap result measures **plausibility against annotations**. Pair it
with controlled model interventions when making faithfulness claims.

## Fixed-prefix token and sequence log probabilities

`fixed_prefix_log_probs(logits, chosen_ids, attention_mask=..., output_mask=...)`
accepts causal logits `[B,T,V]`, explicit teacher-forced token IDs `[B,T]`, and
boolean `[B,T]` masks. It scores token `t` using logits at `t-1`, never at `t`.
The output mask selects only the already-chosen answer tokens; prefix tokens are
excluded. Selected tokens and their immediate predecessors must be unpadded.
Position zero cannot be selected. Left or right padding is supported, but a
selected token immediately following padding is rejected. Valid attention and
position IDs must also have been supplied to the actual model by the caller.

```python
from autoexplain.multimodal_metrics import fixed_prefix_log_probs

# Capture these exact IDs/masks once, then reuse them for all interventions.
def chosen_answer(output):
    return fixed_prefix_log_probs(
        output.logits, teacher_forced_ids,
        attention_mask=nonpadding_mask,
        output_mask=answer_mask,
    )["sequence_log_prob"]
```

Select `.logits`, a dictionary key or a tuple element explicitly in your adapter.
The token sequence passed to the model must match `chosen_ids`. For multimodal
models that expand image placeholders or omit visual positions in language-model
logits, the adapter must first align logits, IDs and masks. This utility does not
infer those positions. Keep the prefix and chosen continuation fixed while
changing evidence, and disable generation/KV caches unless an adapter proves
correct alignment. Seq2seq decoder conventions require a separately aligned
adapter; this is a causal-LM scoring primitive, not a generation API.

Outputs include `token_log_probs` `[B,T]` with zeros outside the selected answer,
`sequence_log_prob` (sum), `mean_log_prob`, `token_count`, and `valid`. An empty
answer has sum zero, mean NaN, and `valid=False`. Unselected IDs may be sentinel
values, but selected IDs must be in vocabulary. Computation preserves float32
and float64 logits; only lower-precision selected logits are promoted to float32.
Autograd is preserved outside a no-grad context; evaluation routines themselves
run without parameter-gradient accumulation.

## Clean/corrupt donor patching

`PatchSite(name, layer, alignment, mask=None)` names an existing PyTorch module.
Sites must be `PatchSite` instances. The module must run **exactly once** per
forward and return a dense floating tensor. Captured and recipient activations
must be finite, even when the downstream score itself would remain finite.
A mask broadcasts to the **full activation including batch**, unlike evidence
group masks. `mask=None` patches the whole output. Each site's required alignment
string documents why clean and corrupt positions correspond; shape equality is
checked but semantic alignment cannot be established automatically.

```python
from autoexplain.multimodal_tracing import PatchSite, donor_patching_sweep

trace = donor_patching_sweep(
    model, clean_inputs, corrupt_inputs,
    sites=[PatchSite(
        "visual_tokens_layer_8", "layers.8.visual_projection",
        alignment="Same batch examples and fixed visual-token positions in both runs",
        mask=visual_token_activation_mask,
    )],
    score_fn=chosen_answer,
)
```

The layer path above is illustrative, not a promised architecture interface.
Use `dict(model.named_modules())` to select a real tensor-output site. A generic
transformer block may return a tuple and must be wrapped or replaced with a
suitable inner tensor-output module. There is no automatic tuple reconstruction,
multimodal token matching, head-axis inference, recurrent call indexing, or
support guarantee for model sharding/offloading, fused attention or KV caches.

For each independent site the evaluator captures clean and corrupt activations,
patches clean into clean and corrupt into corrupt (self-controls), and patches
clean into corrupt (restoration). It returns raw baseline, capture, self-control
and patched scores plus self deltas. Compare capture scores with baseline scores
and self deltas with zero to diagnose stochasticity or an invalid intervention.
It does not silently certify a self-control as passing. Each site is evaluated
independently; this is not a joint causal circuit search.

Recovery is `(patched_corrupt - corrupt) / (clean - corrupt)`. It is not clipped:
negative values, values over one and negative denominators are possible. A
near-zero denominator (`abs <= recovery_epsilon`, default `1e-8`) produces NaN
and `recovery_valid=False`, while raw scores remain available. Nonfinite computed
clean-minus-corrupt denominators, self-patch/restoration differences, or recovery
values at otherwise-valid denominators raise `ValueError`, including float64
overflow despite finite input scores. NaN recovery is reserved for the documented
near-zero-denominator case. No average hides those cases. This does not establish semantic causality without valid corruption,
alignment, score selection and appropriate additional experimental controls.

All forwards use eval/no-grad. Original heterogeneous module training flags are
restored, existing parameter gradients are untouched, hooks are removed even on
errors, and replacements do not modify original activations in place. Do not run
concurrent forwards/hooks on the same model. Models that mutate global state or
non-tensor caches remain the adapter's responsibility. Defaults cap sites at 128,
forwards at 1,024, and each captured activation at 16 million elements. Only two
captured activation clones are retained between forwards; model outputs, masks,
input clones and temporary patch tensors still require memory. Baselines cost
two forwards; each site costs five forwards.

## Bounded descriptive attention payloads

`cross_modal_attention_payload` accepts an iterable of `(layer_name, attention)`
pairs, with each attention tensor shaped `[B,H,Q,K]`. Query and key modality maps
supply explicit integer token indices independently, supporting rectangular
cross-attention as well as self-attention.

```python
from autoexplain.multimodal_tracing import cross_modal_attention_payload

payload = cross_modal_attention_payload(
    selected_layer_attentions,
    query_modalities={"answer": answer_positions},
    key_modalities={"vision": image_positions, "text": prompt_positions},
    include_matrices=False,
)
# payload contains only dictionaries, lists, strings and finite Python numbers;
# json.dumps(payload) is sufficient. No UI framework or file write is performed.
```

Indices must be nonempty, unique and disjoint within each axis's modality map;
query and key maps may of course refer to the same indices in self-attention.
Unlisted padding/special tokens are deliberately omitted, not renormalized away.
Tensors must contain finite nonnegative probabilities with row sums at most
`1 + 1e-4`; masked zero rows are allowed. Obtain these from the model in eval
mode and supply pre-dropout probabilities, not raw logits or post-dropout scaled
weights. Numerical row-sum validation alone cannot certify their origin. Some fused/flash backends do not expose attention weights;
select an appropriate model-supported extraction mode yourself. This function
does not change backend settings or pretend to reconstruct missing attention.

For each modality pair, `mean_key_mass` `[B,H]` sums selected-key probabilities
and averages over selected queries; `mean_pair_weight` divides by selected-key
count. Optional matrices retain `[B,H,Q_selected,K_selected]` values. Every
payload labels these outputs `descriptive_attention_not_causal_attribution`.
Attention mass is not proof that the model relies on the attended content.

Defaults accept at most 32 layers, 16 million input elements per layer and
100,000 total serialized numerical elements, including modality indices, tensor
shapes, summaries and optional matrices. Oversized requests fail rather than
silently truncate. A layer generator avoids retaining additional tensor copies
in the utility; it cannot undo a model's upstream allocation of all attentions.
Selected blocks are processed sequentially and no tensor enters the payload.

## Scope and validation status

These are reusable protocols motivated by grouped evidence evaluation, grounded
video QA, causal tracing and cross-modal attention inspection. They are **not**
complete reproductions of VisFIS, NExT-GQA, object tracing, cross-modal information
flow, or VL-InterpreT: paper-specific training, datasets, architecture mappings,
causal estimators and user interfaces are outside these functions.

This implementation batch received static code/diff review only. No tests,
model runs, benchmarks, dependency installs, downloads or GPU jobs were executed.
Runtime behavior and particular pretrained model integrations remain unverified.
