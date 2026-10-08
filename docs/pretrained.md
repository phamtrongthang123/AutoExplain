# Real pretrained Gemma 4 examples

Notebooks **14 and 15** execute the actual `google/gemma-4-E2B` checkpoint,
pinned to `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`. They complement,
rather than replace, the thirteen offline CPU teaching notebooks. Model weights
are not distributed in this repository. This is a bounded text-only experiment,
not universal model support or a reproduction of a mechanistic-interpretability paper.

## Setup and execution

Use a separate clean Python environment; do not combine this stack with the
older circuit-tracer environment. The measured run used Python 3.12, Transformers
5.18 and matched PyTorch 2.10.0 CUDA 12.6 packages:

```bash
python3 -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[pretrained,jlens,notebooks]'
# Explicitly download the pinned checkpoint into the managed Hugging Face cache:
python scripts/run_gemma4_example.py --allow-download --output /tmp/gemma4-first-run.json
# Subsequent runs are cache-only; the output path must not already exist:
python scripts/run_gemma4_example.py --output /tmp/gemma4-cached-run.json
python scripts/validate_notebooks.py --group pretrained
```

The official checkpoint is approximately **10.25 GB**. Check free disk space,
model terms, available RAM and GPU use before downloading or running. The loader
requires at least 18 GiB available host RAM and checks GPU headroom (at least
6 GiB free, also checked against decoder size). It does not fall back to random
weights, silently download in cache-only mode, use remote code, or quantize.
CPU execution is explicitly selectable but its performance was not benchmarked.

The text model has 4.63 billion parameters, including a 2.35-billion-parameter
per-layer embedding table. That table stays on CPU; gathered rows move to GPU.
The BF16 decoder runs on GPU. Do not call `model.to('cuda')` after split placement.
The loader explicitly maps multimodal checkpoint text keys and rejects missing
or mismatched text weights: merely loading without an exception is insufficient.

## Notebook 14: attribution, residual patching and logit lens

All six baseline capital completions were top-1 correct. Country-token patches
at layer 8 produced the following results (zero-indexed layers):

| Donor → recipient | Desired city | Patched city-pair margin | Actual top-1 | Margin recovery |
| --- | --- | ---: | --- | ---: |
| France → Germany | Paris | 6.75 | Paris | 0.899 |
| Italy → Spain | Rome | 5.50 | ` a` | 0.897 |
| Japan → China | Tokyo | 5.125 | Tokyo | 0.859 |

**The Italy intervention did not make Rome the top prediction.** Recovery is a
ratio of city-pair logit differences, not a probability or success rate. The
notebook retains every predefined case and layer, three norm-matched random
controls, self-patches, and both country- and last-token interventions. Many
later country-token patches had no effect. Final-layer last-token replacement
copies the donor readout and is labeled a trivial positive control, not circuit
discovery. All self-patch logit errors and final logit-lens margin errors were zero.

Gradient-times-activation and gradient norms are reported separately for the
main embedding and per-layer embedding routes. An `inputs_embeds`-only gradient
would omit the second token-identity route. These are local sensitivities, not
complete causal explanations. Plain logit-lens readouts include final normalization
and logit soft-capping.

## Notebook 15: actual Anthropic J-lens

The upstream Jacobian-lens estimator fits layer 8 on four generic sentences,
then evaluates six disjoint capital prompts. It compares the fitted lens,
plain logit lens and a column-permuted fitted matrix. The permutation preserves
matrix norm and singular values while disrupting coordinate alignment.

For the France country token, the fitted lens's top five included `法国`,
` France`, ` францу`, ` Strasbourg` and `France`; the plain lens produced unrelated
tokens. At the last token, the fitted readout included multilingual city-related
tokens, **not Paris**. Country-token ranks and all six readouts are retained in
the notebook and JSON output. This is an input-readability diagnostic, not
next-token accuracy, literal internal thoughts, fit-stability evidence or a
statistical significance test. Other token-identity and shared-KV paths remain
fixed in the source-activation derivative.

## Measured resources and remaining gaps

On an RTX 3080 10 GiB and i7-12700F, cached fresh-kernel execution took:

| Notebook | Wall seconds | Peak host RSS MiB | Peak allocated GPU GiB |
| --- | ---: | ---: | ---: |
| 14: pretrained interventions | 12.544 | 6451.43 | 4.296 |
| 15: pretrained J-lens | 27.102 | 6340.41 | 4.601 |

These observations exclude the initial download and installation; they are not
performance guarantees. Structured outputs are in `examples/outputs/` and executed
notebooks retain tables, plots and measurements. No pretrained SAE dictionary,
Gemma Scope 2 experiment, trained circuit-tracer graph, live Goodfire or live
Codex request is established by these runs. Gemma Scope 2 targets Gemma 3, not
Gemma 4; gated-model terms require an explicit user decision.
