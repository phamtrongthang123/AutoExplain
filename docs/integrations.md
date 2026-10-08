# Tool integrations and local playgrounds

AutoExplain 0.2 prioritizes a broad collection of callable tools. Existing packages
provide many of the algorithms; AutoExplain adds small examples, common entry
points, scoped interventions and explicit support boundaries. An adapter test is
not a reproduction of an upstream paper or validation on a pretrained model.

The separately validated [pretrained Gemma 4 tier](pretrained.md) executes
attribution, patching and actual J-lens on real weights, with retained controls
and failures. Use its separate environment and setup instructions.

## Install only the backends you need

Python 3.12 is recommended for the full collection. The core still targets Python
3.10+, but not every optional upstream stack is tested on every Python version.
Install a matched PyTorch/torchvision CPU or CUDA pair for your environment first.
Do not combine all extras blindly: current circuit-tracer and J-lens require
incompatible Transformers versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[attribution,vision,tabular,diffusion,sae,jlens,openai,goodfire-local,notebooks,playground,docs,dev]'
python -m autoexplain.catalog
python scripts/validate_notebooks.py --group base
python scripts/validate_notebooks.py --group integrations
```

The `jlens` and `openai` extras install code at pinned Git commits and need Git
and internet during installation. Executing the bundled notebooks needs neither
internet nor downloaded pretrained weights. Our tests used CPU PyTorch
2.10.0.dev20251025, Captum 0.9.0, SHAP 0.52.0, Diffusers 0.40.0, Transformers
5.18.0 and SAELens 6.53.0. Dependency bounds are in `pyproject.toml`.

### Separate circuit-tracer environment

```bash
python -m venv .venv-tracing
.venv-tracing/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-tracing/bin/pip install -e '.[tracing,notebooks,dev]'
AUTOEXPLAIN_TRACING_TEST=1 USE_TF=0 .venv-tracing/bin/python -m pytest tests/test_tracing_backend.py -q
.venv-tracing/bin/python scripts/validate_notebooks.py --group tracing
```

The pinned circuit-tracer requires Transformers 4.56–4.57.3 and Hugging Face Hub
below 1. J-lens requires a newer stack. Separate environments are a real dependency
boundary, not an optional performance tweak. Do not add the `sae`, `diffusion`, or
`jlens` extras to the tracing environment.

## Callable method families

| Family | Operations | Entry point |
| --- | --- | --- |
| Captum input attribution | Saliency, Input×Gradient, IG, DeepLIFT, GradientSHAP, DeepLiftSHAP, guided backprop, deconvolution, ablation, permutation, occlusion, LIME, KernelSHAP, sampled/exact Shapley | `backends.captum_attribute` |
| CAM | Grad-CAM, Grad-CAM++, HiResCAM, ScoreCAM, AblationCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, XGradCAM, elementwise Grad-CAM | `backends.cam_attribute` |
| Layer attribution | Layer IG, gradient×activation, conductance, ablation | `backends.captum_layer` |
| SHAP | Tree, permutation, exact, linear explainers | `backends.shap_explain` |
| Evaluation | Infidelity, max sensitivity, deletion/insertion controls, parameter randomization | `backends`, `metrics`, `representations` |
| Concepts/representations | Linear CAV, directional sensitivity, NMF, ridge probe, linear CKA | `concepts`, `representations` |
| Tabular | Permutation importance, PDP/ICE, constrained nearest-candidate counterfactual | `tabular_tools` |
| Language/interventions | Attention rollout, logit lens, capture, patching, feature ablation, concept steering | `representations`, `activations`, `steering` |
| Sparse features | Encode/decode inspection, top examples, error-preserving steering, local SAE load | `features` |
| Diffusion | Matched-noise sampling; layer/step/branch/spatial interventions | `diffusion` |

The catalog counts callable operations/backend variants, not unique scientific
algorithms. Some algorithms appear in both native and upstream implementations.
Exact Shapley is capped at six feature groups; exact SHAP at ten features. Group
image features before using LIME/KernelSHAP. Wrappers keep upstream-specific
arguments explicit rather than pretending every method has identical semantics.

```python
from autoexplain.backends import captum_attribute, cam_attribute

# model: torch.nn.Module returning [B, classes]; batch: preprocessed image tensor
# attribution = captum_attribute(model, batch, method="deeplift", target=0,
#                               baseline=0.0)
# heatmap = cam_attribute(model, batch, layers=["features.3"],
#                        method="gradcam_plus_plus", target=0)
```

DeepLIFT/guided methods require compatible module nonlinearities; arbitrary reused
or functional activations may violate assumptions. Feature permutation requires
multiple examples. GradientSHAP/DeepLiftSHAP require reference distributions.
EigenCAM is class-independent. NMF axes and linear concept separators do not
receive semantic labels automatically. Attention rollout is descriptive, not a
causal attribution. TCAV significance testing is not implemented: the package
provides CAV fitting and a directional-sensitivity statistic, not a complete study.

## Frontier research integrations: actual support boundary

| Source | What ran locally | What did not run / is not claimed |
| --- | --- | --- |
| OpenAI SAEs | Pinned original inference module, tiny synthetic training, encoding/decoding and interventions | Released GPT-2 SAE checkpoint evaluation |
| OpenAI sparse circuits | Actual upstream GPT architecture and activation hook recorder | Learned sparse circuits, training/pruning reproduction or released checkpoints |
| Google ecosystem / SAELens | Actual Standard/JumpReLU SAEs, local save/load, explicit release loader | Gemma Scope 2 pretrained dictionaries or a Gemma base model; access/terms and compute must be checked separately |
| Anthropic J-lens / J-space | Actual reference fitting and readout on a tiny Hugging Face GPT-2 architecture; logit-lens comparison | Paper-level semantic findings, private Claude internals or large-corpus fits |
| Anthropic-origin circuit-tracer | Actual attribution graph on tiny random transformer/transcoders in an isolated environment | Trained semantic circuits or large pretrained graph workloads |
| Goodfire parameter decomposition | Pinned original nano VPD training, all its loss families, two CPU steps | Convergence, published decomposition results or the current JAX trainer's smooth-L0 objective |
| Goodfire Ember | Explicit-opt-in SDK adapter; transport contract tested with a mock | Live service availability, account access or real feature search; archived SDK |

### SAE intervention semantics

`SAEAdapter` supports OpenAI's `(features, normalization_context)` API and
SAELens' tensor API. By default an edit returns
`x + decode(edited_features) - decode(original_features)`, preserving reconstruction
error. This makes a neutral edit an identity operation. Dropping the error term
changes the model even without a feature intervention; test that separately.
The activation site, normalization and feature dictionary must match the original
training configuration. A compatible tensor shape alone is insufficient.

### J-lens and J-space

`fit_reference_jlens` calls Anthropic's actual estimator; `read_reference_jlens`
returns matched Jacobian- and logit-lens readouts. Upstream `jlens.from_hf` mutates
and freezes the supplied model, so use a dedicated instance. The bundled numeric
vocabulary example deliberately makes no semantic interpretation claim.
`fit_local_jacobian` separately exposes a small exact estimator for a caller-defined
causal residual tail. It is not a replacement for an arbitrary model adapter.

### Diffusion

The image tutorial uses a real `diffusers.UNet2DModel` and DDPM scheduler on 8×8
synthetic bars. Interventions explicitly select loop-step indices, tensor-output
layers, batch rows and broadcast spatial masks. Conditional/unconditional batch
ordering is never guessed. The sampler does not implement text encoding, VAE
latents or classifier-free guidance. A complete PixArt/PolypSteer integration
remains outstanding; projection removal alone is not a reproduction.

## References and licenses

Runtime projects: [Captum](https://github.com/pytorch/captum),
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam),
[SHAP](https://github.com/shap/shap), [SAELens](https://github.com/decoderesearch/SAELens),
[Diffusers](https://github.com/huggingface/diffusers).

Research sources: [OpenAI SAEs](https://github.com/openai/sparse_autoencoder),
[OpenAI sparse circuits](https://github.com/openai/circuit_sparsity),
[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-270m-pt),
[Anthropic Jacobian lens](https://github.com/anthropics/jacobian-lens),
[J-space paper](https://transformer-circuits.pub/2026/workspace/index.html),
[circuit-tracer](https://github.com/decoderesearch/circuit-tracer),
[Goodfire parameter decomposition](https://github.com/goodfire-ai/param-decomp),
[Ember SDK](https://github.com/goodfire-ai/goodfire-sdk).

See the repository's `THIRD_PARTY.md` for pinned revisions and vendored license
notices. AutoExplain's MIT license does not override third-party code, checkpoint,
dataset or base-model terms. No third-party pretrained model weights are bundled.
