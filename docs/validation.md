# Validation evidence

## Local verification of version 0.2

The implementation was checked on a Linux i7-12700F machine using Python 3.12.7
and CPU PyTorch 2.10.0.dev20251025. Training and notebook kernels used two threads.
No GPU jobs, external model downloads, paid hosted requests or private datasets
were required for the recorded tutorial executions.

- Main test suite: **105 passed, one intentionally skipped**, 14.29 seconds.
- Isolated circuit-tracer test: **one passed**; its main-environment skip is
  intentional because Transformers dependency requirements conflict.
- All **13 notebooks executed**, with outputs and runtime metadata retained.
- Streamlit AppTest exercised original controls and the optional CAM backend.
- Strict MkDocs build passed.
- Wheel and source distribution built successfully for version 0.2.0.

Warnings include Captum input-gradient/hook notices, Matplotlib/NumPy
compatibility deprecations, and an upstream circuit-tracer backward-hook warning.
The package build also emits a setuptools license-table deprecation notice.
These runs are not evidence of scientific explanation quality or universal
framework compatibility.

## Notebook execution measurements

Wall time includes fresh kernel startup, imports, computation and shutdown.
Peak RSS is the kernel process's lifetime high-water mark, not aggregate process
tree memory, GPU memory, or incremental model memory. Tracing uses a separate
virtual environment. Measurements are observations on this machine, not limits.

| Notebook | Wall seconds | Peak kernel RSS MiB |
| --- | ---: | ---: |
| 01 CNN attribution | 2.760 | 402.79 |
| 02 U-Net segmentation | 2.776 | 406.04 |
| 03 Transformer patching | 2.387 | 366.45 |
| 04 Toy diffusion | 2.792 | 392.62 |
| 05 Tree paths | 2.288 | 393.70 |
| 06 Attribution backends | 4.366 | 665.80 |
| 07 SAE features | 6.110 | 648.90 |
| 08 Reference J-lens | 5.337 | 605.35 |
| 09 OpenAI sparse circuits | 1.851 | 276.82 |
| 10 Goodfire nano decomposition | 3.406 | 446.02 |
| 11 Diffusers image steering | 8.382 | 700.34 |
| 12 Representation/tabular controls | 2.484 | 409.74 |
| 13 Circuit tracing | 6.548 | 715.55 |

## What the tests establish

Tests check analytic gradient/IG properties, a known Grad-CAM calculation,
LayerCAM's local weighting, hooks and mode restoration, exact projection-removal
behavior, zero-strength identity controls, SHAP/backend output contracts, SAE
local serialization, neutral feature edits, linear Jacobian recovery, CKA
invariances and real upstream execution on small fixtures. All 15 Captum input
variants, four layer variants, 11 CAM variants and four SHAP explainer families
are invoked in tests, rather than only registered as names.

The integration notebooks exercise actual external code for J-lens,
circuit-tracer, OpenAI sparse models and Goodfire nano decomposition. Random
models/transcoders and two-step training validate execution only; they cannot
establish the semantic interpretations or convergence reported in research papers.
The image diffusion notebook runs an actual Diffusers U-Net, but its small bars
experiment is not text-to-image generation or a PolypSteer reproduction.

## External capabilities not validated

- Google Gemma Scope 2 pretrained artifacts and base models: loader API exists,
  but no real artifact download/evaluation was performed.
- Goodfire Ember live API: opt-in wrapper and mocked transport checks only;
  service enrollment/availability is unverified.
- Codex GPT-6-Luna: model identifier exists in the local CLI cache; command/schema
  tests are mocked. No successful live request is claimed by this report.
- Full PixArt/PolypSteer intervention policy, pretrained language-model semantic
  findings, complete TCAV significance studies, and arbitrary-model automatic
  compatibility are not established.

## Re-run

Follow [integration setup](integrations.md), then:

```bash
USE_TF=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES='' python -m pytest -q
python scripts/validate_notebooks.py --group base
python scripts/validate_notebooks.py --group integrations
# In the isolated tracing environment:
AUTOEXPLAIN_TRACING_TEST=1 USE_TF=0 python -m pytest tests/test_tracing_backend.py -q
python scripts/validate_notebooks.py --group tracing
```

Dependency installation requires network access; tutorial execution does not.
Use a fresh notebook kernel before running all cells, because PyTorch inter-op
thread count cannot be repeatedly reset after parallel work begins.
