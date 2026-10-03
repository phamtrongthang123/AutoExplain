# Thirteen offline CPU tutorial notebooks

These playgrounds use small synthetic datasets and either tiny trained models or explicitly labeled random fixtures.
They require no network, checkpoints, API keys, or external datasets. They are
teaching examples, not pretrained models, clinical tools, or paper reproductions.

## New integration notebooks

Use the [integration setup](integrations.md) for optional dependencies. Run notebooks 06–12 with `python scripts/validate_notebooks.py --group integrations`. Run notebook 13 in the separate tracing environment with `python scripts/validate_notebooks.py --group tracing`.

| Notebook | Executed workflow |
| --- | --- |
| `06_attribution_backends.ipynb` | Captum IG, multiple upstream CAM variants, actual TreeSHAP |
| `07_sae_features.ipynb` | OpenAI SAE training/inspection/intervention and SAELens JumpReLU local round-trip |
| `08_jacobian_lens.ipynb` | Actual Anthropic J-lens fitting and logit-lens comparison on tiny HF GPT-2 |
| `09_openai_sparse_circuits.ipynb` | Actual OpenAI sparse GPT architecture and activation recorder |
| `10_goodfire_parameter_decomposition.ipynb` | Actual nano VPD training, target preservation and component ablation |
| `11_diffusers_image_steering.ipynb` | Actual 8×8 image U-Net, matched-noise and random-direction controls |
| `12_representation_and_tabular_controls.ipynb` | CKA, probe, CAV, NMF, permutation, PDP/ICE and candidate counterfactual |
| `13_circuit_tracing.ipynb` | Actual graph attribution with random tiny model/transcoders; isolated environment |

All eight execute without network access or pretrained model files. Read each notebook's opening cell: several intentionally use random weights, and no notebook is presented as a reproduction of a frontier model's scientific findings.

## Run and retain outputs

From the repository root, use an environment containing `torch`, `numpy`,
`scikit-learn`, `matplotlib`, `matplotlib-inline`, `nbformat`, `nbclient`,
`jupyter-client`, and `ipykernel`. Tests additionally require `pytest`.

```bash
python -m pip install -e '.[notebooks]'
python scripts/validate_notebooks.py
python -m pytest tests/test_adapters.py tests/test_toys.py tests/test_tabular.py
```

The validator runs the notebooks sequentially in fresh kernels using its own
Python interpreter, with a 180-second **per-cell** timeout. It overwrites each
successfully executed notebook with its outputs and execution metadata; an error
stops validation rather than fabricating results. It prints one JSON measurement
record per notebook. It does not install packages or change shared kernelspecs.
For interactive use, open a notebook from the repository root or `notebooks/`,
and restart its kernel before rerunning all cells (PyTorch inter-op thread count
can only be set once per process).

## Coverage and assumptions

| Notebook | Model and input | Explicit explanation target |
| --- | --- | --- |
| [CNN](https://github.com/phamtrongthang123/AutoExplain/blob/main/notebooks/01_cnn_attribution.ipynb) | Two-convolution classifier, `[B,1,16,16]` | Signed input derivative and Grad-CAM of a selected class logit |
| [Segmentation](https://github.com/phamtrongthang123/AutoExplain/blob/main/notebooks/02_unet_segmentation.ipynb) | One-level skip-connected U-Net-like model, `[B,1,16,16]` to `[B,2,16,16]` | Mean foreground logit inside a fixed weighted ROI |
| [Transformer](https://github.com/phamtrongthang123/AutoExplain/blob/main/notebooks/03_transformer_patching.ipynb) | One bidirectional encoder block, integer `[B,6]` tokens | Embedding gradients `[B,6,8]`, exact donor-token embedding patch, self-patch control |
| [Diffusion](https://github.com/phamtrongthang123/AutoExplain/blob/main/notebooks/04_diffusion_steering.ipynb) | Two-dimensional noise predictor with normalized scalar timestep | Matched-initial-noise sampling under positive, negative, zero, and random hidden-direction interventions |
| [Tree](https://github.com/phamtrongthang123/AutoExplain/blob/main/notebooks/05_tree_tabular.ipynb) | Shallow single-output sklearn classifier, three numeric features | Exact branch predicates, leaf, predicted class, and class probabilities |

`SegmentationScoreAdapter` accepts a tensor-returning model and an optional fixed
finite nonnegative `[H,W]` ROI with positive mass. Its output is `[B,C]` weighted
**mean logits**, without softmax. ROI shape must match output spatial dimensions;
structured segmentation outputs need another explicit wrapper.

`patch_token` expects a tensor-output module with exactly matching donor and
recipient `[B,T,D]` shapes. It replaces one token across the matched batch and
removes its hook even on exceptions. Do not share a hooked model concurrently.
The example patches embeddings, not an identified internal reasoning circuit.

`explain_tree_path` supports fitted single-output `DecisionTreeClassifier` only.
It converts the row to float32, matching sklearn's prediction input precision.
Nonfinite values, mismatched features, multi-output trees, regressors, and forest
ensembles are not supported. The result is not SHAP or a causal attribution.

## Reproducibility and measurement

Each notebook fixes NumPy and PyTorch seeds, uses CPU tensors, and bounds PyTorch
intra-op and inter-op threads to two. The validator also bounds common BLAS/OpenMP
thread environment variables to two before launching kernels and hides CUDA.
Datasets contain at most 160 rows/examples; training loops have at most 90 steps.
Identical numerical results across PyTorch versions or CPU architectures are not
guaranteed. No accuracy threshold is presented as a generalization benchmark.

The last cell records two measurements:

- `tutorial_compute_wall_seconds`: elapsed time after imports/thread setup through
  tutorial computation and plotting; excludes kernel launch and initial imports.
- `kernel_lifetime_peak_rss_mib`: the kernel's own `RUSAGE_SELF.ru_maxrss` high-water
  mark (Linux KiB, macOS bytes converted to MiB). Includes imports, but excludes
  child processes and the validator. It is not incremental model memory, GPU
  memory, total system memory, or aggregate child-process RSS.

The validator adds `execution_wall_seconds`, including kernel startup, imports,
execution, and shutdown, but excluding the final notebook-file write. Outputs and
`metadata.autoexplain_execution` retain the evidence for an actual run.
`resource` measurement targets Linux/macOS; Windows requires a different memory
measurement implementation.

All thirteen notebooks were executed and saved with outputs on this machine. The retained execution measurements are listed in [Validation](validation.md); they are local observations, not performance guarantees.

## Interpretation limits and references

- Gradients are local sensitivities; normalized Grad-CAM can hide magnitude and
  sign. [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391).
- The compact segmentation architecture illustrates a skip connection, not a
  full implementation of [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597).
- The transformer uses no causal mask, padding, natural language, or pretraining.
  [Vaswani et al.](https://arxiv.org/abs/1706.03762) motivate the architecture;
  [Meng et al.](https://arxiv.org/abs/2202.05262) provide intervention background,
  not a method reproduced by this one-token embedding replacement.
- Toy diffusion uses the epsilon-prediction training objective and a deterministic
  DDIM-style update. Its 12-step beta schedule ends far from pure noise, so normal
  initial samples are deliberately approximate. Hidden directions are fit at one
  timestep but applied across all steps; directional meaning may not transfer.
  [Ho et al.](https://arxiv.org/abs/2006.11239),
  [Song et al.](https://arxiv.org/abs/2010.02502). This is not PolypSteer.
- Tree paths describe exact computation but do not establish causality; impurity
  importance is not local attribution and can be biased. See the
  [scikit-learn tree guide](https://scikit-learn.org/stable/modules/tree.html).
