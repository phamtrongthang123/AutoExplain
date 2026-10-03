# Third-party code and research provenance

AutoExplain-authored code is MIT-licensed. Dependencies and bundled third-party
files retain their original licenses; the root MIT license does not replace them.
No third-party pretrained weights, patient data, or private datasets are included.
Scientific citations are in the method and integration documentation.

## Vendored code

| File | Original source and pinned revision | License / modifications |
| --- | --- | --- |
| `src/autoexplain/_vendor/openai_sae.py` | [OpenAI sparse_autoencoder/model.py](https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/model.py), `4965b941e9eb590b00b253a2c406db1e1b193942` | Original MIT text in `_vendor/OPENAI_SAE_LICENSE.txt`; provenance header added, inference implementation otherwise unchanged |
| `src/autoexplain/_vendor/goodfire_nano.py` | [Goodfire nano_param_decomp/run.py](https://github.com/goodfire-ai/param-decomp/blob/017a17af4f7372d0db8e55b23aeff603ee6ba735/nano_param_decomp/run.py), `017a17af4f7372d0db8e55b23aeff603ee6ba735` | Original MIT text in `_vendor/GOODFIRE_LICENSE.txt`; provenance header and `typing.override` fallback for Python before 3.12 added |

The OpenAI inference module is vendored because the original package pins Torch
2.1.0 and TransformerLens 1.9.1, conflicting with modern optional backends. This
avoids silently bypassing its training-package dependency requirements. The
training code and pretrained checkpoint loaders are not vendored.

The Goodfire nano implementation is explicitly a reference L_p VPD objective;
it is not interchangeable with the current JAX trainer. Its W&B dependency is
imported only when that optional module is requested; AutoExplain's bounded
wrapper disables logging and does not initialize a remote run.

## Pinned optional source installations

- [Anthropic Jacobian lens](https://github.com/anthropics/jacobian-lens), Apache-2.0:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- [OpenAI circuit sparsity](https://github.com/openai/circuit_sparsity), Apache-2.0:
  `dbf1fe0d27b76c19e10d2a715f28c2e5da535e08`.
- [Decode Research circuit-tracer](https://github.com/decoderesearch/circuit-tracer), MIT:
  `7f66876689f59e92fc3641650d38b8bd41749ec4`.
- [Goodfire Ember SDK](https://github.com/goodfire-ai/goodfire-sdk), Apache-2.0:
  PyPI version `0.3.5`; upstream repository is archived. Hosted service access was
  not verified. This is not Goodfire's server implementation or a model release.

Other optional backends include Captum (BSD-3-Clause), pytorch-grad-cam (MIT),
SHAP (MIT), scikit-learn (BSD-3-Clause), SAELens (MIT), Transformers and Diffusers
(Apache-2.0). Consult the installed release's own license and notices; their source
is not bundled here. Version bounds are declared in `pyproject.toml`.

## Artifact and service boundaries

Google Gemma Scope 2's official artifact card declares CC-BY-4.0; underlying Gemma
models have separate terms. Neither is redistributed here. The SAELens loader
requires explicit download opt-in, and the user must handle any license/access
requirements. OpenAI SAE/circuit checkpoint licensing must be reviewed for the
exact artifact before redistribution; a repository code license alone is not
sufficient evidence for every externally hosted file.

API access does not grant access to a provider's private internal activations.
OpenAI, Anthropic, Google, and Goodfire names identify research provenance, not
endorsement or affiliation. No trademark license is asserted.
