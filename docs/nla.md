# Natural Language Autoencoders (NLA)

AutoExplain supports **pretrained activation → natural language → activation**
round trips for the released **Qwen2.5-7B-Instruct, block 20** NLA pair. This is
not sparse autoencoding, J-lens transport, or asking a generic LLM to label a
vector. The AV and AR are separately fine-tuned models with a discrete natural
language bottleneck.

## Sources and implementation boundary

Primary paper: Fraser-Taliente, Kantamneni, Ong et al.,
[Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html),
Transformer Circuits, 2026. The Method, NLA training/inference appendix, poetry
steering appendix, and open-model appendix establish the architecture,
reconstruction objective, normalization and limits of interventions.

Official implementation inspected at immutable revision
[`0577769b55ad4fdd96d159e983361b97fa4e7331`](https://github.com/kitft/natural_language_autoencoders/tree/0577769b55ad4fdd96d159e983361b97fa4e7331):

- [`nla_inference.py`](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/nla_inference.py): AV prompt/embedding injection, sidecar validation, AR reconstruction.
- [`nla/models.py`](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/nla/models.py): truncated AR backbone, removed final norm, separate bias-free value head.
- [`nla/loss.py`](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/nla/loss.py): per-example normalized MSE.
- [`nla/datagen/extractors.py`](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/nla/datagen/extractors.py): zero-based **block output** extraction convention.

Released artifacts:

- AV: [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av)
- AR: [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar)
- [AV sidecar](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av/blob/main/nla_meta.yaml),
  [AR sidecar](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar/blob/main/nla_meta.yaml)
  and both `config.json` files were inspected directly. These artifact URLs
  refer to mutable `main`, not a verified checkpoint checksum; pin an artifact
  revision and retain checksums in your own deployment manifest.

The adapter executes these **released model/checkpoint contracts using native
Hugging Face Transformers**, entirely locally. Unlike the upstream inference
client, it loads the full AV and calls HF `generate(inputs_embeds=...)` instead
of sending embeddings to an SGLang HTTP service. It does not import upstream
source or use the upstream loaders' `trust_remote_code=True`. This alternate
execution path has been reviewed against the source, **not empirically validated
against SGLang or real checkpoints**. No model was loaded during implementation.

The official code is Apache-2.0 (Copyright 2026 Anthropic PBC); see its
[LICENSE](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/LICENSE).
This adapter binds to the published data/model contracts without vendoring the
upstream implementation. Checkpoint and base-model terms remain independent of
AutoExplain's license. Consult checkpoint license/NOTICE files before use or
redistribution; other upstream model families include Gemma/Llama restrictions.
The paper's Claude models are not released through this integration.

## Prerequisites: explicit, local, large

In a dedicated environment, install the optional dependencies with
`python -m pip install -e '.[nla]'` from the repository. This command is provided
for the caller; it was not run during implementation. The extra supplies
`transformers==4.57.1`, `safetensors` and `PyYAML` alongside core `torch`.
For the connected interface, install `.[studio]` instead; it includes these NLA
dependencies plus the UI and resource-inspection packages.
The released Qwen configs record Transformers **4.57.1**; this is the upstream
reference version, not a claim of an AutoExplain-tested dependency matrix.
Transformers must provide native Qwen2 and embeddings-only generation support.
If another AutoExplain extra requires a different Transformers version, prefer
a separate environment until that combination has been validated. SGLang,
Miles, API keys and the full training repository are **not required** here.

Obtain the above AV/AR checkpoints and the original
`Qwen/Qwen2.5-7B-Instruct` source model separately under their applicable terms.
The core NLA API never installs packages or fetches artifacts. The Studio page
has a separate, explicitly authorized foreground preparation action for the
three fixed repositories; it never fetches at startup. Changing the selected
artifact or its requested revision clears download consent; authorize the new
selection explicitly. Each NLA directory needs:

- `config.json`, tokenizer/chat-template assets and `nla_meta.yaml`;
- complete local `model.safetensors` or indexed safetensors shards;
- for AR, **`value_head.safetensors`**, containing the trained `weight` tensor.
  A random or missing head is not an NLA reconstructor.

Supply materialized checkpoint directories or complete configured Hub-cache
snapshots. Snapshot symlinks are allowed only when the resolved snapshot is
`HF_HUB_CACHE/models--<expected-repository>/snapshots/<40-hex-commit>` and each
link resolves to a regular, hash-named blob in that same repository's `blobs`
directory. Symlinked repository/snapshots/blobs directories, arbitrary external
links, directory links, escaping shard index paths and missing shards are
rejected. Validation covers metadata/tokenizer assets as well as weights. This
reuses existing Hub assets without copying tens of GB. Pickle-only weights,
quantized checkpoints, Hub IDs passed as local paths, remote code and incomplete
snapshots are rejected. All HF loads specify
`local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True` for
models. Sidecar YAML uses safe loading. This does not authenticate the weight
files: obtain them from a trusted source and verify their provenance yourself.

`NaturalLanguageAutoencoder(...)` is inert. For API compatibility, `.load()`
retains the original explicit two-model residency and `encode()`, `decode()`,
`reconstruct()` and `edit()` contracts; that legacy path does not impose the new
staged preflight. CPU requires float32. CUDA permits float32/float16/bfloat16 and
checks bfloat16 support. There is no sharding, offload or quantization. Both
models plus a source model can exceed ordinary workstation RAM/VRAM; this legacy
example is **not** the recommended lifetime pattern on a 10 GiB GPU.

For bounded residency, use `.load_av()` → `.encode()` → `.load_ar()` → `.decode()`.
Each staged load first releases the previous NLA model. `.release()` explicitly
releases weights/tokenizers, collects garbage and clears unused CUDA allocator
blocks when applicable; it never deletes checkpoint files. Defaults remain CPU
float32. A conservative additional available-memory estimate is 42 GiB for
source/AV and 34 GiB for AR, including loading/workspace headroom. CUDA stage
loads additionally check free device memory. Available RAM is read via psutil or
Linux `/proc/meminfo`; unknown/insufficient RAM refuses loading unless the caller
explicitly passes `override_memory=True`. These estimates are not guarantees,
do not reserve RAM, and cannot account for every loader/runtime allocation.
An OOM never triggers automatic quantization, reduced precision or retries.

```python
nla = NaturalLanguageAutoencoder(av_path, ar_path)  # CPU float32
try:
    # The caller must first release its source model; acts retains CPU rows only.
    nla.load_av(progress=print)  # optional cancel=lambda: cancellation_requested
    texts = nla.encode(acts)
    nla.load_ar(progress=print)  # AV released before AR loading begins
    prediction = nla.decode([text.explanation for text in texts])
    replacement = nla.edit(acts, [text.explanation for text in texts], edited)
finally:
    nla.release()
# Only now reload the source to run the same-input intervention comparison.
```

Stage progress callbacks receive a stage string on the calling thread. Optional
`cancel` callbacks return true to raise `InterruptedError`; loaders release on
failure/cancellation, and inference callers should always use `finally` to
release. Boundaries occur before/after stage loading, each AV generated token,
and each AR explanation. A native loader or one full forward cannot be
interrupted internally by these callbacks.

## Connected Studio workflow

`autoexplain.studio_nla.render()` provides the self-contained NLA page. Its caller
must unload the ordinary Studio model before entering this page. The page keeps
no models in shared caches or session state and uses these lifetimes:

1. Explicitly prepare each fixed source/AV/AR repository, or supply local paths
   and validate all three without loading model weights. **Advanced · checkpoint
   paths and revisions** defaults to collapsed on every render, including first
   run. Artifact selection, download consent, Prepare, resource guidance and
   readiness/validation controls remain visible outside Advanced. Revisions
   default to mutable `main`; users can pin commits in Advanced. Preparation
   always targets the configured Hub cache, not an Advanced local-directory
   override. Complete local snapshots are reused first. No credentials are
   collected in the page; configure Hub authentication outside it if needed.
2. Tokenize source text locally without a chat template or silent truncation.
   The page bounds source text to 4,000 characters / 128 tokens, exposes actual
   token IDs and indices, and selects 1–4 source positions. Choose an exact
   single-token next-token target, or lock the original source argmax.
3. Load source CPU float32, extract block-20 rows and score the locked baseline;
   release source. Load AV, generate complete explanations (32–256-token budget
   per selected row), release AV. Load AR, reconstruct directions and compute
   normalized MSE/cosine, then release AR.
4. Edit each explanation (at most 3,000 characters, with the AR's 512-token
   prompt limit enforced rather than truncated). Decode edited descriptions
   with AR, retain the original reconstruction residual, release AR and reload
   source. Compare the identical input IDs and locked next-token target across
   unpatched baseline, unchanged self-patch, norm-matched reconstruction and
   residual-preserving edit. Report target logit/probability, argmax token ID
   and target-logit deltas. This is not cached answer generation or a whole-answer
   faithfulness claim. Leaving explanation text unchanged supplies the
   unchanged-text intervention control.

The page shows current available RAM and free space on the selected artifact's
actual Hub-cache blob filesystem, refreshed on page rerun without loading
weights or contacting the Hub. Disk inspection follows the configured
`HF_HUB_CACHE/models--<repository>/blobs` destination; if it does not exist yet,
its nearest existing ancestor supplies the filesystem reading. Thus a cache or
repository/blob mount need not share the home filesystem. Failed resource
readings display **unknown**, not a fabricated capacity.

Before starting a transfer, `prepare_artifact()` first tries to validate and
reuse a complete local snapshot. Only when it needs preparation does it recheck
free disk space against a conservative remaining-download upper bound: **36 GiB
for source, 36 GiB for AV, 28 GiB for AR**, including metadata/transfer headroom.
Incomplete snapshots and partial blobs receive no space credit, because their
sizes do not establish reusable contents; this can refuse a resumable download
that would actually fit. Unknown/insufficient free space refuses the transfer
with a path-free actionable message unless the user explicitly checks the
visible **Override low/unknown cache-disk preflight** control (API:
`override_disk=True`). That override is separate from download consent and the
RAM override. Already complete validated snapshots do not require extra free
space or a disk override. Budgets are not live remote artifact sizes, reserve no
space, and cannot guarantee a fit or account for other processes' allocations.

Artifact preparation uses an owned, foreground Hub subprocess, fixed repository
allowlisting and native Hub cache/resume behavior. Streamlit **Stop** reaches a
main-thread progress callback approximately every 250 ms during transfer and
terminates/reaps the child, retaining partial blobs for a later explicit retry.
Stage loading and each source/AR forward may take minutes before the next Stop
boundary; AV generation checks between tokens. No instant inference cancellation
is promised. All loading/inference operations release model references in
`finally`, including native Streamlit interruption. There is no detached work.

The page retains one current bounded result and at most eight numerical history
records per browser session. Input/token selection/budget changes invalidate the
current result; edited text/strength changes invalidate comparison results.
Checkpoint selection changes clear results/history, and a file-size/mtime stamp
is checked before execution to reject changed assets. That stamp is mutation
detection, **not** cryptographic provenance. Unload/clear removes NLA session
state without deleting checkpoint files. Exports are allowlisted numerical
records only: no source text, explanation text, paths, credentials or requested
revisions. Upstream exception text is not displayed. Native framework logs and
external processes remain outside this export policy.

This workflow has been **statically reviewed only**. No downloads, launches,
inference, tests, builds or installations were performed during implementation;
real-checkpoint behavior, peak memory and cancellation timing remain unverified.

## Runnable local API example

Replace the three paths below with complete local directories. This example
performs real inference when run by the caller; it is not a bundled download or
an assertion that inference was run during implementation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from autoexplain.core import evaluating
from autoexplain.nla import (
    NaturalLanguageAutoencoder, extract_nla_activations,
    patch_nla_activations, nla_reconstruction_loss,
)

source_path = "/models/Qwen2.5-7B-Instruct"
av_path = "/models/nla-qwen2.5-7b-L20-av"
ar_path = "/models/nla-qwen2.5-7b-L20-ar"
device, dtype = "cuda:0", torch.bfloat16  # select hardware with enough memory
local = dict(local_files_only=True, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(source_path, **local)
source = AutoModelForCausalLM.from_pretrained(
    source_path, torch_dtype=dtype, use_safetensors=True, **local,
).to(device).eval()
ids = tokenizer("The rabbit hopped into the garden.", return_tensors="pt")["input_ids"].to(device)

# One unpadded sequence. Every index refers to this exact source tokenization.
acts = extract_nla_activations(source, ids, positions=[ids.shape[1] - 1])
nla = NaturalLanguageAutoencoder(av_path, ar_path, device=device, dtype=dtype).load()
round_trip = nla.reconstruct(acts)
print(round_trip.texts[0].explanation)
print(round_trip.direction_mse, round_trip.cosine_similarity)

# The individual encoding and decoding operations are also public.
texts = nla.encode(acts)
raw_prediction = nla.decode([item.explanation for item in texts])
loss_per_row = nla_reconstruction_loss(raw_prediction, acts.vectors)

# Supply a deliberate edited explanation, rather than changing source text.
original = [round_trip.texts[0].explanation]
edited = [original[0].replace("rabbit", "mouse")]
replacement = nla.edit(acts, original, edited, strength=0.5)
# If the explanation did not contain "rabbit", this edit has no effect.
# Run baseline and intervention on IDENTICAL input IDs; score a chosen outcome.
with evaluating(source), torch.inference_mode():
    baseline_logits = source(input_ids=ids, use_cache=False).logits
    with patch_nla_activations(source, acts, replacement.to(device=device, dtype=dtype)):
        edited_logits = source(input_ids=ids, use_cache=False).logits
# Compare a prespecified next-token/behavioral metric; do not treat text as proof.
```

## Activation, text and reconstruction contracts

- **Layer:** zero-based `source.model.layers[20]` **output**, after its residual
  additions, width 3584. This is HF `hidden_states[21]`, not `[20]`. The upstream
  README's abbreviated example uses `[20]`; this adapter follows the explicit
  extractor and AR implementation, which keep blocks 0 through 20 inclusive.
- **Source:** the original Qwen2.5-7B-Instruct weights, not the fine-tuned AV.
  Architecture and user-supplied metadata are checked, but cannot authenticate
  weight identity. Other finetunes and layer transfers are not supported claims.
- **Shape/alignment:** `NLAActivations.vectors` is finite floating `[N,3584]`;
  `token_ids` is the complete single unpadded source sequence and `positions`
  selects unique increasing indices, one per vector. Special/chat tokens count
  as tokens; character spans, padding and batched sequences are not inferred.
  Vectors must have nonzero norm. AV descriptions are one per selected source
  token, not a token-by-token alignment between explanations and source text.
- **AV:** sidecar chat prompt and tokenizer are used verbatim. The injection
  character and both neighboring token IDs must match. The vector is normalized
  to L2 norm **150** and replaces exactly one prompt embedding. There is no
  learned input projection. Qwen embeddings need no Gemma-style scaling.
- **Generation:** a fresh configuration fixes greedy decoding, one beam, no
  repetition penalty, and a configurable 1–512-token budget (default 200).
  It intentionally differs from the paper's temperature-1 sampling and is not
  intended to reproduce paper metrics. No seeded sampling or arbitrary
  generation overrides are exposed. Greedy does not guarantee bitwise equality
  across devices/libraries. Missing, empty or ambiguous explanation tags raise
  an error rather than silently accepting truncated text. Results retain raw
  text, generated IDs, source position and source token ID.
- **AR:** 21-layer checkpoint, no final RMSNorm, no language-model head, and the
  released bias-free `Linear(3584,3584)` value head. Uses the non-chat sidecar
  template with `add_special_tokens=True`; validates BOS policy and final suffix
  IDs. Reads the final real token; overlong text raises instead of truncating.
- **Reconstruction:** `decode()` returns raw CPU float32 vectors. Their norms
  are not calibrated source norms. `reconstruct()` normalizes prediction and
  target to `sqrt(3584)`, returns both, their full difference (target minus
  prediction), per-row MSE and cosine similarity. At this scale,
  `MSE = 2*(1-cosine)`. Raw norm fidelity is not measured. Near-zero vectors are
  rejected rather than assigning a misleading direction score.
- **State:** inference uses temporary evaluation/inference contexts; source
  extraction restores all module train/eval flags and removes hooks in `finally`.
  Patch contexts do not change train/eval flags and remove hooks even on errors.
  They enforce one full uncached forward, exact output dtype/device and identical
  source token IDs. Supply `use_cache=False` and all forward arguments other than
  `input_ids` by keyword; cached state and embedding-only inputs are rejected
  before execution. They are not concurrent-safe.

## Diagnostics, interventions and limits

`nla_reconstruction_loss` exposes only the differentiable normalized AR MSE
component. The paper jointly trains AV (RL, open-model GRPO with KL regularization)
and AR (supervised reconstruction), following summarization SFT warm starts.
The adapter **does not reproduce SFT, GRPO, distributed training, reward shaping,
dataset construction, optimizer schedules or published evaluation results**.

FVE is not `1-MSE` in general. To compute it, compare average normalized squared
error with the matching normalized activation distribution's mean-prediction
baseline variance: `1 - mean(error^2) / mean((target - population_mean)^2)`.
Use a representative held-out corpus and a separately estimated reference mean,
not one selected token. FVE is undefined for zero baseline variance. The API
therefore reports MSE and cosine rather than inventing an FVE denominator.

`edit()` is an explicit **error-preserving decode-based intervention primitive**:

```
x' = x + strength * ||x|| * (unit(AR(edited_text)) - unit(AR(original_text)))
```

At strength 1, it substitutes a norm-matched edited reconstruction while retaining
`x - ||x|| * unit(AR(original_text))`. This is an AutoExplain residual-preserving
variant of the paper's decode/edit interventions, **not an exact reproduction of
its steering experiments**. It deliberately does not renormalize the final sum,
which would alter the retained error. Use the explanation corresponding to each
selected activation. `patch_nla_activations` installs the returned vectors at
those source positions without modifying weights; it is not a generation hook.

Low reconstruction MSE establishes directional information recoverable by the
AR, not semantic truth, disentanglement or causal importance. NLA descriptions
can confabulate, omit information or exploit shared AV/AR priors. Edited text may
be out of distribution and change multiple concepts at once. Measure actual
model-output effects against the same-input baseline, include unchanged-text
and reconstruction controls, and monitor degeneration. The paper reports cases
where direct reconstruction patching does not produce the desired behavior.

**Not implemented:** Gemma-3/Llama released variants, Claude variants, arbitrary
activation layers, multimodal inputs, server-backed SGLang inference, batch
padding, cached generation interventions, training reproduction, quantized or
sharded loading. No tests, installs, inference/training runs, checkpoint downloads
or empirical quality checks were performed for this implementation; validation
was limited to primary-source and static code review.
