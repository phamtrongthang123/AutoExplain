# Feature-level MM-SHAP and InterSHAP

`autoexplain.feature_shap` implements joint feature Shapley games and the
published MM-SHAP and InterSHAP aggregation equations. These are algorithmic
components, not reproductions of all pretrained models, masking configurations,
background distributions, datasets or reported experimental results.

## Feature players and replacements

```python
from autoexplain.multimodal import MultimodalScorer, modality_shapley
from autoexplain.feature_shap import feature_shapley, mm_shap, intershap

# Fixed output target; image and token IDs share a batch dimension.
scorer = MultimodalScorer(
    lambda parts: model(parts["image"], parts["tokens"])[:, fixed_class],
    model=model,
)
features = feature_shapley(
    scorer, {"image": images, "tokens": token_ids},
    {"image": masked_images, "tokens": masked_token_ids},
    feature_masks={"image": image_feature_ids, "tokens": text_feature_ids},
    feature_modalities=["vision", "vision", "text", "text"],
    method="exact",
)
contributions = mm_shap(features.contributions, features.feature_modalities)

# InterSHAP uses a separate game whose players are entire modalities.
modalities = modality_shapley(
    scorer, {"image": images, "tokens": token_ids},
    {"image": masked_images, "tokens": masked_token_ids},
)
interaction_summary = intershap(modalities.interactions)
```

Masks contain global feature IDs `0..G-1`; `-1` leaves an element fixed. In this
example image IDs use 0/1 and token IDs use 2/3. Masks broadcast to **full input
shapes including batch**. For `[B,C,H,W]`, `[H,W]` groups pixels across channels.
Per-sample layouts can use `[B,1,H,W]`. Every declared feature must appear in at
least one mask; an absent per-sample feature is a dummy player for that sample.
A single ID can group dependent fields such as token IDs and attention-mask
positions. All fields with that ID change together. The supplied modality label
assigns the whole feature to one semantic modality; the library cannot establish
whether the grouping is semantically appropriate.

References name exactly the mask keys, match dtype/device and have full input
shape or singleton batch. Other fields and `-1` elements remain original even in
the empty coalition. Thus completeness is relative to this explicit feature
universe, not necessarily to an entirely blank image or sentence. Replacements,
fixed targets and batch independence follow `MultimodalScorer`'s contract.

The paper uses text `[MASK]` replacements, zeroed image patches and a policy
balancing image/text feature granularity. Callers must implement appropriate
preprocessing; zero in normalized image space is not necessarily a black pixel.
This API does not infer special tokens or model-specific attention masks.

## Exact and sampled feature Shapley

`method="exact"` enumerates all `2**G` coalitions and computes the standard
weighted marginal contributions. Defaults allow 64 declared features but cap
exact coalitions at 4,096, so the exact default permits at most 12 features.
The scored-row budget is checked before any model execution. Score storage is
`O(B * 2**G)`, in addition to caller inputs/masks.

`method="permutation"` averages marginal contributions over independent random
feature orders (default 128; at least 2 required). It uses at most
`B * (2 + permutations * (G-1))` scored rows, reusing empty/full endpoints, and
`O(B*G)` estimator storage. It does not cache exponentially many coalitions.
Pass a `torch.Generator` to own randomness; its device is used for permutations.
Orders are shared across batch rows, so errors across examples are correlated.
Returned per-feature standard errors use the sample variance across orders and
measure Monte Carlo error only. They are not confidence intervals for model
faithfulness, baseline selection, or a dataset-level statistic.

Both methods return signed contributions, baseline/full scores, efficiency
residuals and estimator metadata. Exact results also retain coalition scores;
permutation results retain standard errors. The permutation estimator is a
standard Shapley estimator, **not** a claim to reproduce the paper's particular
coalition-sampling configuration. Finite arithmetic overflow raises an error.

## MM-SHAP: absolute before aggregation

For signed feature Shapley values `phi[b,g]`, equations (2)-(3) define
`mass[b,m] = sum(g in m, abs(phi[b,g]))` and
`share[b,m] = mass[b,m] / sum_m mass[b,m]`.

`mm_shap` implements these equations for any number of modalities. Taking the
absolute value after summing features is incorrect for this metric. Likewise,
whole-modality Shapley values are not interchangeable with feature-level values.
The result contains modality names, absolute masses, sample-level shares and
mean shares across explicitly valid samples. An all-zero row has NaN shares,
`valid=False`, and is excluded from that mean; `valid_count` reports the retained
sample count. This undefined-case policy is a documented library extension.

## InterSHAP: global and local are different

`intershap` accepts `[B,M,M]` symmetric SHAP interaction matrices, with half of
each pair effect in each off-diagonal entry and remaining main effects on the
diagonal. `modality_shapley(...).interactions` uses this convention.

The primary paper distinguishes:

- **Global:** `Phi = abs(mean_batch(I))`, then
  `sum_off_diagonal(Phi) / sum_all(Phi)` (equations 5-8).
- **Local:** the same ratio for `abs(I[b])` per sample, followed by the mean of
  sample ratios (equations 10-11).
- **Individual modality contributions:** diagonal mass divided by total mass
  (equation 9), not row sums that include interactions.

Global signed effects may cancel before taking absolute values. The global
score is therefore not the mean local score or a ratio of mean absolute effects.
Both off-diagonal halves are counted. The implementation returns both variants,
modality contributions, validity flags and the global absolute-mean matrix.
Zero-denominator scores are NaN; the reported local mean explicitly excludes
invalid rows and includes `local_valid_count`. Output/class reduction is not
inferred: use consistent scalar objectives across the dataset. Reproducing a
multiclass experiment requires its explicit class/background aggregation policy.

## Primary-source provenance and status

Definitions were read directly from the supplied primary publication PDFs:

- [MM-SHAP, ACL 2023](https://aclanthology.org/2023.acl-long.223.pdf), section 3.3,
  equations 2-3, and masking discussion immediately following;
  [official code](https://github.com/Heidelberg-NLP/MM-SHAP).
- [InterSHAP, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/35452/37607),
  sections 3.1-3.4, equations 2-11;
  [official code](https://github.com/LauraWenderoth/InterSHAP).

Code here is an independent implementation of the stated equations, not copied
upstream code. Only static source/diff review has been performed. No tests,
model runs, benchmarks or paper-result reproductions have been executed.
