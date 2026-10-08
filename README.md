# AutoExplain

**Give a model an explanation workflow—and test what happens when you intervene.**

AutoExplain is an early-stage local explainability app and Python toolkit for capability-based interpretability and activation steering. Its long-term goal is to support multiple model families through explicit adapters. Version 0.2 combines native methods with Captum, CAM, SHAP, SAE, Jacobian-lens, circuit-tracing, Goodfire decomposition and Diffusers integrations. It does **not** support arbitrary AI models automatically; each backend has explicit requirements and a tested scope.

[Documentation](https://phamtrongthang123.github.io/AutoExplain/) · [Runnable playground guide](docs/playground.md) · [PolypSteer integration notes](docs/diffusion.md)

## Long-term vision

AutoExplain will keep growing as the research field grows. My ambition is to eventually use it to explain and investigate any model—from LLMs, diffusion models, CNNs, U-Nets, decision trees, BERT and JEPA-style models to systems used in robotics, healthcare and other domains. Reaching that goal will require model-specific methods, explicit assumptions and evidence about where explanations and steering work or fail. It is a research ambition, not a claim of universal support today.

The package is also a tool for pursuing my own scientific questions. The [Scientific Roadmap](ScientificRoadmap.md) sets out the five- and ten-year questions that guide its development: causal control in neural surrogates, knowledge recovered from expert interaction, selective generalization of corrections, and the effects of AI delegation on human capability. I want the growing collection of runnable experiments to help answer those questions, rather than merely expand a catalogue of explanation methods.

## Install and run

Requires Python 3.10+ and PyTorch 2.1+. No API keys, external datasets, or model downloads are needed for the synthetic demo.

```bash
python -m pip install "autoexplain-ai @ git+https://github.com/phamtrongthang123/AutoExplain.git"
python -m autoexplain.demo
```

The package is distributed from GitHub, **not currently published on PyPI**. For the local AutoExplain app, use a clean environment:

```bash
git clone https://github.com/phamtrongthang123/AutoExplain.git
cd AutoExplain
python -m venv .venv-studio
source .venv-studio/bin/activate
python -m pip install -e '.[studio]'
autoexplain-studio
# Equivalent: python -m autoexplain.studio_cli
```

Open `http://127.0.0.1:8501`. The studio opens on **recorded real Gemma 4 pipelines**: exact prompts, measured next-token predictions, signed attributions and controlled donor patches, with inputs and outputs side by side. No weights load on startup. A persistent sidebar separates these examples from the live text, image and NLA workspaces; technical settings stay secondary. Saved reports are bundled with the package and labeled recorded, with source hashes and original model/environment metadata—not passed off as fresh inference. An explicit bounded rerun uses only the pinned cached checkpoint in a compatible Transformers-5 interpreter; it never installs or downloads anything.

- **Text:** SmolLM2-135M continuations, readable signed token highlights, and side-by-side original/edited prompts explaining the same next-token target.
- **Images:** CLIP candidate-text scores and crop-aligned pixel integrated-gradients overlays with adjustable opacity/color map.
- **Activations in language:** the actual Qwen2.5-7B block-20 NLA pair, with explicit checkpoint preparation, token selection, verbalization, reconstruction diagnostics, description editing and controlled source-model comparisons. CPU float32 stages release one model before loading the next; these large models can be very slow and require substantial available RAM.
- **Real example pipeline:** France → Germany donor patching changes the recorded next token from ` Berlin` to ` Paris`; self/random controls and the failed Italy patch remain visible. These are next-token measurements, not invented full answers. **Real CNN images** uses ImageNet-pretrained ResNet-18 on an uploaded photograph, with actual class predictions, Grad-CAM and input gradients; no synthetic training or fallback.

Named text/image history retains eight runs per session; NLA retains eight numerical records. Explicit **Save experiment** writes a bounded local record; **Saved experiments** provides Open/export/selected Delete, separate from transient session history. Records include input text and results in unencrypted server-local storage—review the privacy notice before saving. **Use this example** opens editable, native-token-constrained Gemma inputs in the same pipeline. Progress and Stop operate at documented boundaries—not in the middle of a tensor operation. See the [local app guide](docs/playground.md) for resource estimates, privacy and exact limits, and the [UX reference notes](docs/studio-ux.md) for the design basis.

This studio revision includes [sequential Chromium UX reviews](docs/studio-ux-review.md) with manual navigation, readiness and local-record interactions, not automated tests or new model inference. Live inference/cancellation remain unverified because cache/resource prerequisites were not met; the bundled Gemma reports are from prior real runs, not newly executed checks. The `studio` extra includes NLA dependencies and pins Transformers 4.57.1. Use `.venv-studio` (Transformers 4.57.1) for the app's text/image/NLA adapters. Keep `.venv-llm` (Transformers 5.18) separate for Gemma reruns; the runner selects that dedicated backend without changing the server environment. The general `.venv` also contains Transformers 5.18 and should not be used to launch the text/image/NLA workspace. The real CNN page also needs torchvision matched to PyTorch (already present in this checkout's Studio environment; available through the `[vision]` extra); official weights download only after explicit consent and Prepare. The headless synthetic toolkit demo remains separate. GitHub Pages serves documentation, not the local Python application.

## Bring your model

```python
from autoexplain import Inspector

# your_model is an instantiated torch.nn.Module.
# batch is your correctly preprocessed floating tensor, on the model's device.
inspector = Inspector(your_model, checkpoint="weights.pt")
for recommendation in inspector.suggest():
    print(recommendation)

# Choose a candidate after checking its role in your architecture.
heatmap = inspector.gradcam(batch, layer="features.2", target=0)
signed_gradient = inspector.input_gradients(batch, target=0)
```

A checkpoint cannot reliably identify an architecture or its preprocessing. You provide the model, input transformation, and task semantics. The checkpoint loader accepts a state dict or a dictionary containing `state_dict`, uses `weights_only=True`, and requires an exact architecture match. Never load untrusted checkpoints into a privileged process.

## What works today?

| Model or task | Implemented | Requirement / limitation |
| --- | --- | --- |
| PyTorch CNN classification | Grad-CAM, LayerCAM, input gradients, integrated gradients, SmoothGrad, occlusion | Explicit class scores and appropriate layer |
| Other differentiable PyTorch classifiers | Gradient/path/noise attribution and perturbation controls | Floating inputs, independent batch samples and differentiable scores |
| Tensor-output modules | Capture, masked donor patching, additive steering and positive-projection removal | Explicit hook site, alignment, feature axis and direction |
| Segmentation and tiny transformer tutorials | Explicit score/embedding workflows | Synthetic task adapters, not universal model support |
| Decision-tree/tabular tutorial | Tree paths and perturbation explanations | Optional scikit-learn extra; see tutorial API |
| Diffusers image U-Net | Matched-noise sampling, timestep/layer/branch/spatial interventions | Tiny image workflow tested; no full PixArt/PolypSteer reproduction |
| LLM research tools | Actual J-lens, OpenAI sparse-circuit architecture, SAEs, circuit-tracer and Goodfire nano decomposition | Small random/synthetic fixtures; pretrained research findings not reproduced |
| Pretrained Gemma 4 E2B | Two-route gradients, residual patching, logit lens and actual fitted J-lens | Pinned text-only checkpoint; six illustrative prompts, not universal support |
| Other pretrained LLMs/BERT/JEPA | Not validated | Architecture-level interfaces are not blanket pretrained-model support |
| TensorFlow, JAX | Not implemented | Framework adapters remain future work |

Recommendations are structural heuristics with prerequisites, not evidence of scientific suitability. The package combines established attribution baselines with modern research implementations. It makes no state-of-the-art performance claim. Captum supplies 15 input and four layer-attribution variants; pytorch-grad-cam supplies 11 CAM variants; SHAP supplies four explainer families. Automatic semantic labeling and universal method selection are not implemented.

## Multimodal implementation batch (static review only)

New torch-only workflows support explicit named-modality scoring, exact modality
Shapley interactions, scalar EMAP, feature-level Shapley with MM-SHAP aggregation,
and distinct global/local InterSHAP scores. CLIP tools include explicit similarity
targets and a source-grounded single-layer Grad-ECLIP primitive requiring
single-head forward intermediates—not an automatic pretrained encoder adapter.
Additional utilities cover SpLiCE-style sparse coding, grouped evidence
perturbations, spatial/temporal grounding overlap, fixed-prefix token scoring,
controlled donor patching and bounded attention payloads.

See [contribution contracts](docs/multimodal-contributions.md),
[MM-SHAP and InterSHAP](docs/feature-shap.md),
[CLIP and concepts](docs/multimodal-concepts.md), and
[evaluation and tracing](docs/multimodal-protocols.md). These additions received
**static source/diff review only; no tests or model runs were executed**.
They do not establish full coverage of the 133-paper bibliography, reproduce all
paper pipelines, or make generic SAE and attribution tools equivalent to named
research methods. Model-specific adapters, training pipelines and remaining
paper methods are still unfinished.

## Natural Language Autoencoders (NLA)

[The NLA integration](docs/nla.md) implements local activation → natural language
→ reconstructed activation inference for the released **Qwen2.5-7B-Instruct,
block-20** verbalizer/reconstructor pair. It includes normalized reconstruction
diagnostics and residual-preserving text-edit interventions. This is distinct
from SAE features and J-lens. Complete local AV/AR checkpoints are required;
loading is explicit and never downloads weights automatically.

Use `[studio]` for the connected interface or `[nla]` for the API alone in a
separate environment: both pin the checkpoint reference Transformers version,
which conflicts with this repository's Transformers-5 extras. This native-HF execution path received **source/static review only**:
no inference or tests have been run. Other NLA model families and SFT/GRPO
training reproduction are not implemented. See the guide for memory, model
license, exact activation-layer and tokenizer requirements.

## Real pretrained examples and thirteen offline CPU notebooks

Two additional executed notebooks use **real Gemma 4 E2B**: [14: interventions](notebooks/14_gemma4_pretrained.ipynb) and [15: actual J-lens](notebooks/15_gemma4_jacobian_lens.ipynb). Six baseline completions were correct; layer-8 country patches switched the top prediction in two of three pairs (Italy produced ` a`, not Rome). J-lens uses separate fitting/evaluation prompts and a scrambled-matrix control. See [setup, results and limitations](docs/pretrained.md). These require cached weights and a separate CUDA environment; they are not offline-from-scratch CPU tutorials.

The original five tutorials cover a CNN, tiny U-Net, tiny transformer, toy diffusion process and decision tree. Eight additional notebooks exercise upstream attribution libraries, SAE features, actual Anthropic J-lens, OpenAI sparse-circuit code, Goodfire nano parameter decomposition, a real Diffusers image U-Net, representation/tabular controls, and actual circuit-tracer graphs.

```bash
# Install only the extras you need; full commands and a separate tracing env are documented.
python -m pip install -e '.[attribution,vision,tabular,notebooks]'
python scripts/validate_notebooks.py --group base
python -m autoexplain.catalog  # discover callable operations and limitations
```

See [integration setup and backend contracts](docs/integrations.md), the [notebook guide](docs/notebooks.md), and [validation evidence](docs/validation.md). J-lens and circuit-tracer currently require **separate environments**. All thirteen notebooks have been executed locally on CPU, without downloading model weights; none requires hosted API access.

## Broad optional tool collection

- **Captum:** DeepLIFT, GradientSHAP, DeepLiftSHAP, guided backprop, feature ablation/permutation, LIME, KernelSHAP, Shapley sampling, layer conductance and more.
- **CAM:** Grad-CAM++, HiResCAM, ScoreCAM, AblationCAM, EigenCAM, FullGrad and other variants.
- **Concepts and representations:** CAV fitting, directional sensitivity, NMF, CKA, linear probes, attention rollout and logit-lens readouts.
- **Sparse features and circuits:** OpenAI SAE code, SAELens, J-lens/J-space readouts, OpenAI sparse-model inspection, circuit-tracer graphs and Goodfire nano VPD.
- **Tabular and evaluation:** SHAP, PDP/ICE, candidate counterfactuals, infidelity, sensitivity, perturbation controls and parameter randomization.

**Boundaries:** Gemma Scope 2 pretrained artifacts and live Goodfire Ember access have not been executed here. Their loader/remote interfaces are labeled accordingly. A two-step decomposition or random-transcoder graph validates execution, not semantic quality. See [third-party provenance and licenses](THIRD_PARTY.md).

## Codex-assisted research workflow

An optional structured planner uses `codex exec` with **`gpt-6-luna` / `high`**. Review its JSON proposal, then run the allowlisted experiment locally. No generated Python is executed by the local runner. Model weights and datasets are not attached to prompts by AutoExplain; Codex read-only sandboxing is not a confidentiality boundary.

```bash
python -m autoexplain.research_cli plan-demo --question "Which pixels affect the vertical-bar score?" --out proposed-plan.json
# Inspect the proposal before execution.
python -m autoexplain.research_cli run-demo --plan proposed-plan.json --out experiment-001
```

The local runner saves measurements, controls and attribution tensors. This initial workflow is CPU classification only; it is not yet an autonomous research loop. CLI calls consume your Codex quota. See [configuration, privacy limits and validation status](docs/research.md).

## Steering

```python
from autoexplain import concept_direction, steer

# Separate, matched reference activations of shape [observations, features].
v = concept_direction(positive_activations, negative_activations)
with steer(your_model, "features", v, feature_axis=1, strength=0.5):
    changed_scores = your_model(batch)
# The hook is removed even if the forward pass raises.
```

For PolypSteer-inspired suppression, select `mode="remove_positive_projection"`. That primitive implements `h' = h - strength * max(<h,v>, 0) * v` for a normalized direction. It does not reproduce PolypSteer's model-, timestep-, or conditioning-specific intervention policy.

## Research context

[Mechanist](https://github.com/zjunlp/Mechanist) motivates automated interpretability investigation; AutoExplain currently provides a deterministic toolkit rather than an autonomous research agent. No Mechanist source is included.

[PolypSteer: Counterfactual Endoscopic Synthesis via Training-Free Activation Steering](https://arxiv.org/abs/2603.07066), by Trong-Thang Pham, Loc Nguyen, Anh Nguyen, Hien V. Nguyen, and Ngan Le, motivates the diffusion-steering direction. Use the [official implementation](https://github.com/UARK-AICV/PolypSteer) for paper reproduction. No pretrained medical model or patient data is bundled here.

Grad-CAM follows [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391). Heatmaps can be misleading and should be checked with randomization, perturbation and task-specific tests. An intervention changing a score does not establish semantic or clinical validity.

## Development

```bash
python -m pip install -e '.[dev,docs,playground,notebooks]'
pytest -q
python -m autoexplain.demo
mkdocs serve
python -m build
```

Further work includes pretrained-artifact validation, full text-conditioned PixArt/PolypSteer integration, richer feature discovery and additional framework adapters. Tool breadth and small runnable examples take priority over a fully autonomous research loop. The scientific questions in the roadmap remain unchanged.

MIT licensed, accountless and local. The core has no telemetry or paywall; optional paid services are only a future possibility. Contributions should include numerical tests, explicit supported shapes and tasks, and a runnable example without private data.
