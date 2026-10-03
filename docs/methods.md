# Attribution and activation interventions

These methods explain a chosen model score under explicit input or activation changes. Their outputs do not establish causal importance in the data-generating process. Compare explanations against numerical checks and matched controls before interpreting them.

## Shared contract

Import attribution functions from `autoexplain.attribution`, activation contexts from `autoexplain.activations`, and evaluation from `autoexplain.metrics`.

Attribution and evaluation take a PyTorch `model`, finite floating `inputs` shaped `[B, ...]`, and `target=None`, an integer class index, or an integer tensor/list `[B]`. The model must return `[B, classes]` scores. Wrap structured outputs explicitly. `target=None` selects the clean-input argmax once and holds that class fixed across all perturbations. Scores can be logits or probabilities, but their interpretation differs; the package applies no softmax.

Samples must be independent across the batch. Gradient methods differentiate the sum of selected scores, which does not yield independent per-sample gradients for batch-coupled models. Inputs are cloned; existing parameter `.grad` values are not changed. Attribution and evaluation temporarily enter evaluation mode and restore every module's original training flag, including on exceptions. They do not restore arbitrary mutable model buffers or side effects. Gradient methods require a differentiable path and work inside `torch.no_grad()`, but not `torch.inference_mode()`.

No API changes global CPU thread counts. For CPU tutorials or coordinator verification, set `torch.set_num_threads(2)` in the calling process. Random generators must be compatible with the input device. Hook-based APIs must not share a model across concurrent requests.

## Integrated gradients

```python
integrated_gradients(model, inputs, target=None, *, baseline=0.0, steps=64)
```

Returns `IntegratedGradientsResult(attributions, completeness_delta)`, with attribution shape equal to inputs and delta `[B]`. `baseline` is a finite scalar or tensor broadcastable to inputs. `steps` is a positive integer number of integration intervals, using `steps + 1` gradient evaluations and trapezoidal endpoint weights.

For baseline `b`, the attribution is `(x-b)` multiplied by the numerical integral of the input gradient along `b + alpha*(x-b)`. Delta is `sum(attributions) - (score(x)-score(b))`; a small delta checks numerical completeness, not explanation validity. Increase steps to assess convergence. Nonsmooth paths and saturation can require more steps. Baseline choice determines the question being asked.

Reference: Sundararajan, Taly and Yan, [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017.

## SmoothGrad

```python
smoothgrad(model, inputs, target=None, *, samples=32, noise_std=0.1, generator=None)
```

Returns mean signed gradients with input shape. Each sample adds independent Gaussian noise with standard deviation `noise_std` in absolute input units. This is neither squared-gradient SmoothGrad nor gradient-times-input. No clipping or range-relative scaling is applied. Choose noise after accounting for preprocessing, and pass a seeded `torch.Generator` for reproducibility. Smoothing can reduce visual noise without making an explanation more faithful.

Reference: Smilkov et al., [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825), 2017.

## Occlusion

```python
occlusion(model, inputs, target=None, *, baseline=0.0,
          window_shape=None, strides=None)
```

Returns signed score drops with input shape. Window and stride tuples cover **all nonbatch dimensions**, including channels; defaults are one in each dimension. For `[B,C,H,W]`, use `(C,h,w)` to mask all channels within a spatial patch. Require `1 <= stride <= window <= dimension`. The last window is anchored to each far boundary to ensure full coverage. Overlapping windows contribute their full score drop to each covered scalar, and the result averages those drops. It is not an additive decomposition and generally does not satisfy completeness.

Each window needs one forward pass. Large windows can be expensive and introduce out-of-distribution inputs. Zero is only a useful reference when zero has a meaningful interpretation in the input representation.

Reference: Zeiler and Fergus, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), ECCV 2014.

## LayerCAM

```python
layercam(model, inputs, target=None, *, layer, normalize=True)
```

Requires `[B,C,H,W]` inputs and an explicit named module returning a 4D activation exactly once per forward. Returns `[B,H,W]`. Before bilinear resizing, the map is `relu(sum_channels(activation * relu(local_gradient)))`. Unlike Grad-CAM, gradients are not spatially averaged. Downstream in-place activations are isolated by cloning hook outputs. Hooks are removed even when forward or validation fails.

Default min-max normalization is per image; constant maps become zero. Set `normalize=False` to retain unnormalized, nonnegative maps. Normalized maps do not support absolute magnitude comparisons across examples. This implementation handles a single selected layer, not the paper's multilayer fusion procedure.

Reference: Jiang et al., [LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](https://doi.org/10.1109/TIP.2021.3089943), IEEE TIP 2021.

## Activation capture and patching

```python
with capture_activations(model, layers) as activations:
    output = model(inputs)

with patch_activation(model, layer, replacement, mask=None):
    patched_output = model(inputs)
```

`layers` is a string or a nonempty sequence of unique module names. Capture yields `{name: [tensor_per_invocation]}`. Snapshots are detached clones, so later in-place operations cannot overwrite them. Repeated layer execution and multiple forwards are retained in execution order. Outputs must be tensors; the context retains snapshots until the caller releases them.

`replacement` must exactly match the layer output's shape, dtype and device. It is detached and cloned on context entry. An optional boolean `mask` broadcasts to the output shape and replaces only selected positions. The output is replaced without mutating the original tensor. Repeated invocations receive the same replacement, so the caller must ensure semantic and shape alignment. Full replacement blocks upstream gradients; masked replacement retains gradients only in unpatched positions.

Both contexts remove their hooks on normal or exceptional exit and leave train/eval flags unchanged. They do not themselves run a forward or suppress autograd. Capture donor activations under an explicit evaluation/no-grad context when that is the intended experiment. Existing hook registration order matters: a capture hook registered before patching records the unpatched output. These primitives do not automatically perform causal tracing, donor matching, token alignment or statistical evaluation.

Related application: Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), NeurIPS 2022. The context API is a generic intervention primitive, not a reproduction of that protocol.

## Perturbation faithfulness and random controls

```python
perturbation_faithfulness(model, inputs, attributions, target=None, *,
                         mode="deletion", baseline=0.0, steps=20,
                         random_trials=5, generator=None)
```

Attributions must be finite and have exactly the input shape. Features are individual scalars, ranked by descending absolute attribution separately for each example; ties preserve flattened order. Thus RGB channels are ranked separately. Supply another evaluator if the experiment requires spatial groups, signed rankings or tokens.

Deletion replaces ranked features with baseline values, starting from the original input. Insertion restores ranked features into the baseline. Both include unchanged and fully changed endpoints. With `F` scalar features, `K = min(steps,F)+1` points are evaluated at rounded, evenly spaced feature counts. Integration uses the actual changed fractions rather than assuming exact uniform spacing.

`FaithfulnessResult` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `fractions` | `[K]` | Fraction of scalar features replaced/restored |
| `scores` | `[B,K]` | Raw selected-class scores |
| `auc` | `[B]` | Trapezoidal area against changed fraction |
| `random_scores` | `[R,B,K]` | Curves for independent random permutations |
| `random_auc` | `[R,B]` | Corresponding random-control areas |

`R=random_trials` must be positive. Random controls use the same baseline, targets, counts and perturbation operation. In a conventional positive-evidence setting, lower deletion AUC and higher insertion AUC suggest better ranking relative to random. Absolute-value ranking includes negative evidence, so that direction is not universal. Raw-score AUC is not bounded to `[0,1]`, is sensitive to score scaling, and should not be compared across different models without justification. Report control variability; random controls are not confidence intervals or a guarantee against perturbation artifacts.

Reference: Petsiuk, Das and Saenko, [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421), BMVC 2018, for deletion/insertion evaluation. The scalar-feature implementation here is not the RISE explanation generator or a reproduction of its blurred-image insertion baseline.

## Verification scope

`tests/test_attribution.py` checks linear and quadratic IG completeness, linear SmoothGrad, overlapping occlusion, LayerCAM's local-gradient formula, fixed targets, mode restoration and hook cleanup. `tests/test_activations.py` checks snapshots, repeated invocation, masked patching, nested hook order and failure cleanup. `tests/test_metrics.py` checks exact deletion/insertion curves and AUC, seeded controls and baseline endpoints.

The coordinator independently ran the scoped tests with `.venv/bin/python -m pytest`: **19 passed in 0.95s**. The scope comprised `tests/test_attribution.py`, `tests/test_activations.py`, and `tests/test_metrics.py`. The implementation worker did not execute tests; this result records the coordinator's reported verification.
