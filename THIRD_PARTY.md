# Third-party code and research provenance

AutoExplain-authored code is MIT-licensed. Dependencies and bundled third-party
files retain their original licenses; the root MIT license does not replace them.
No third-party pretrained weights, patient data, or private datasets are included.
Scientific citations are in the method and integration documentation.

## Bundled example photograph

`src/autoexplain/_examples/astronaut.png` is the real NASA photograph of astronaut
Eileen Collins, copied unchanged from the installed scikit-image sample data.
Scikit-image's `data.astronaut` documentation identifies the
[NASA Great Images source](https://flic.kr/p/r9qvLn) and states: “No known copyright
restrictions, released into the public domain.” This photograph is not generated
imagery, a model output, or an endorsement. It is included as an optional real-image
input; no scikit-image runtime dependency is required.

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

## Natural Language Autoencoders integration

`src/autoexplain/nla.py` independently implements the released NLA checkpoint
contracts using Transformers; it does not vendor or import the official training
repository. Primary paper: [Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html).
Official source inspected: [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders/tree/0577769b55ad4fdd96d159e983361b97fa4e7331),
revision `0577769b55ad4fdd96d159e983361b97fa4e7331`, Apache-2.0 (Anthropic PBC).

The optional integration requires separately obtained Qwen2.5-7B-Instruct L20
AV/AR checkpoints and the source model. No weights are bundled or downloaded.
Their license/NOTICE terms must be checked independently; AutoExplain's MIT
license does not relicense them. Artifact sidecars were inspected on mutable
Hub `main`, not authenticated by weight checksums. Native-HF inference has not
been empirically compared with upstream SGLang. See [the NLA guide](docs/nla.md)
for exact artifacts, source files and supported contracts.

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
