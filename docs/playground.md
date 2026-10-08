# AutoExplain local studio

The studio is a local **explain → intervene → compare** workspace, not a hosted
chat/training service. Its default workspace shows **recorded real Gemma 4 E2B
pipelines**, not a synthetic demo or a setup form: exact inputs, measured next-token
outputs, signed attributions and donor-patching controls. A persistent sidebar
keeps real text/image and NLA workflows available. No weights are bundled or
loaded on startup, and nothing downloads automatically.

## Install and launch

Use a clean environment, then install from this repository:

```bash
python -m pip install -e '.[studio]'
python -m autoexplain.studio_cli
# Optional: python -m autoexplain.studio_cli --port 8502
```

Open `http://127.0.0.1:8501` locally or `http://<server-tailscale-ip>:8501`
from an authorized Tailscale device. The packaged launcher stays in the foreground,
binds to **0.0.0.0 (all IPv4 interfaces)**, disables Streamlit usage telemetry and caps uploads
at 10 MiB. Stop the server with **Ctrl+C**. It does not provision a service,
container or cloud inference endpoint. The console entrypoint is
`autoexplain-studio` when installed with the updated package metadata.

The original checkout command remains available:

```bash
streamlit run playground/app.py --server.address=0.0.0.0 --browser.gatherUsageStats=false --server.maxUploadSize=10
```

Use `[studio]` for the pretrained workflows. The real CNN page additionally
requires torchvision matched to PyTorch (available through `[vision]`, already
present in this checkout's Studio environment). Studio uses native Transformers **4.57.1**, Hub
**0.34–0.36** (`>=0.34,<1`), safetensors, Pillow, Streamlit, matplotlib, psutil and PyYAML (for NLA sidecars).
Use a separate environment from the Transformers-5 pretrained/diffusion/SAE
extras. These installation commands are instructions for the caller, not a claim
that installation or runtime validation was performed during implementation.

## Recorded real pipelines → explicit rerun

The landing page reads packaged copies of `gemma4_pretrained.json` and
`gemma4_jlens.json`, independent of the launch directory. The original reports
remain unchanged. Each saved report is labeled **recorded real run**, with its
SHA-256 and checkpoint revision. The intervention report exposes its recorded
dependency/hardware metadata; the J-lens report lacks those fields and is labeled
accordingly. No completion text, execution date or new measurement is invented.

Start with the France donor and Germany recipient: the recipient predicts
` Berlin`; the recorded layer-8 changed-country-token patch predicts ` Paris`
with a Paris-versus-Berlin logit margin of 6.75. The interface retains all three
country pairs, including the Italy patch whose top token is ` a`, **not Rome**.
Inspect signed embedding-route and per-layer-input-route attributions, the patch
position/layer, and self/random controls. Final-layer readout controls are labeled
trivial controls, not discoveries of a semantic mechanism. A next-token output
is not a whole generated answer; a city-pair logit margin is not a probability.
The separately recorded J-lens report remains evidence of its prior run, not a
claim that the bounded patch rerun also refits J-lens.

Choose **Use this example** to copy the selected real pair into **Edit & run**.
Donor/recipient prompts and target/foil text are editable; the native tokenizer
requires 2–32 tokens per prompt, equal lengths, exactly one changed token before
the final position, and distinct single-token targets. Validation precedes weight
loading; no truncation or arbitrary code is accepted. The rerun is an explicit user
action with supported layer choices. It uses the existing pretrained API and the pinned local Gemma snapshot
with offline/cache-only loading, never an automatic install, download, arbitrary
shell command or alternate checkpoint. Interpreter, dependency, cache and resource
readiness are shown before work. Native Streamlit **Stop** terminates and reaps
the owned subprocess; it is never detached. Failed/cancelled attempts are not
replaced with the saved report and labeled new. Inspect the live-run controls for
exact prerequisites. Each run is limited to one aligned prompt pair, one of
layers 0/8/16/24/34, both patch positions, three random seeds and route gradients:
180 seconds and 1 MiB output. Backend checks have a 30-second limit. The known
checkout backend is `.venv-llm/bin/python`, with Transformers 5.18 and matched
PyTorch 2.10.0 / torchvision 0.25.0 / torchaudio 2.10.0 CUDA 12.6 builds. If no
checkout backend exists, the current interpreter must pass the same metadata gate;
no incompatible fallback runs. Checks require BF16 CUDA, 18 GiB available host RAM
and 6 GiB free GPU memory, then the loader checks decoder headroom again. These
checks are not fit guarantees. A process lock serializes example jobs; unrelated
GPU jobs are not controlled, and free-memory readings are only snapshots.

Use the dedicated `.venv-studio` environment for `[studio]` / `[nla]` with
Transformers **4.57.1**, and keep `.venv-llm` with Transformers **5.18** for Gemma.
Recorded data does not import Transformers. Gemma work uses its separately checked
Transformers-5 interpreter; this does not validate text/image/NLA adapters under
5.18. Start the server from `.venv-studio` to enable those adapters. The app reports
an incompatible server runtime and keeps affected workflows disabled with
guidance; it never modifies environments itself.

## Ask → prepare → explain → compare

1. Use the sidebar to choose text, image, or activation-in-language work after
   inspecting the recorded pipeline. **Real CNN images** accepts an uploaded
   photograph and explains an actual ImageNet-pretrained ResNet-18 prediction.
2. Open **Models** for supported workflow cards, then select a model. The app distinguishes assets available locally from
   a model loaded in this session, recommends device/dtype from current resources,
   and shows approximate fit guidance. Revision, local directory and dtype/device
   overrides are under **Diagnostics & local snapshot**. Review the model card/license, then use
   **Prepare**. Complete local snapshots are reused; missing Hub assets require
   explicit download consent. Existing Hub authentication is reused if present;
   never enter credentials in the app.
3. A visible activity indicator, elapsed-time updates and file-level progress
   (where supported) remain active during transfer. They are **not a byte
   percentage or completion estimate**. Use
   Streamlit's top-right **Stop** to cancel: the owned downloader subprocess is
   terminated and reaped, not detached. Partial Hub blobs remain for a later
   **Prepare** retry using the Hub's supported cache/resume behavior. Hard
   termination of the entire server/OS is outside graceful Stop handling.
4. Preparation then loads the local safe weights. Leave the Advanced path empty
   to reuse the cached revision, or supply a trusted complete local snapshot.
   Local-directory loading never fetches missing files. CPU requires float32;
   CUDA-0 supports float32/float16/bfloat16 (hardware support required). Preparing
   a replacement first releases this session's previous model; failure leaves no
   replacement loaded. Cache readiness checks required files, not weight integrity;
   the native loader still rejects missing or mismatched parameters.
5. Enter inputs, then explicitly click **Explain and compare text** or **Explain
   image**. Editing controls alone does not run pretrained inference.
6. Inspect scores and explanation data; **Export this explanation JSON** exports locally.
   Changed inputs/model/settings retain the previous completed result with a stale notice. Changing
   library settings requires another Prepare before execution.
7. Click **Unload session model** to release this session's model; completed results remain available. Downloads
   remain cached; nothing deletes cache files. Other sessions are untouched.
   The NLA workspace releases the ordinary studio model before its staged work.
   Close/unload unused sessions to avoid duplicates.

### Exact supported model/method pairs

| Release | Connected workflow | Approximate budget |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M` | Native Llama causal-LM greedy continuation; signed input-embedding gradient × input for a fixed next-token raw logit; original/edited-prompt comparison | Reserve ~0.6 GB cache; roughly 1–3 GB CPU RAM or 1–2 GB CUDA VRAM |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | Native Transformers CLIP image/candidate-text scoring; existing pixel integrated gradients with magnitude overlay | Reserve ~0.7 GB cache; roughly 2–4 GB CPU RAM or 1–3 GB CUDA VRAM |

These are planning estimates, **not measured peaks or guarantees**. Weight-only
memory is roughly parameter count × dtype bytes (~135M and ~151M parameters);
Python/framework state, gradients, activations, CUDA allocator reservations and
loading buffers add substantial overhead. CUDA still needs CPU RAM. Cache
revisions, incomplete blobs and tokenizers add disk usage. Resource display uses
the home filesystem; check the actual filesystem yourself if `HF_HOME` redirects
the cache. There is no quantization, automatic device fallback, sharding or offload.

Downloads allow only tokenizer/config/text and safetensors assets. Model loading
uses `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`
and explicit native model classes with supported architecture checks. Pickle-only
snapshots, arbitrary model classes and uploaded Python/model code are unsupported.
Local snapshots are **not cryptographically authenticated** as the named release;
use trusted files. Model/config identity and resolved cached commit (or an
explicit unverified-local label) appear in result metadata. A resolved commit is
reported only for a 40-character lowercase hexadecimal snapshot name inside the
configured Hub cache's directory for the selected repository. A directory merely
named `snapshots/<name>` is not sufficient. This path metadata does not authenticate
weights, even for a recognized cache location. Offline Load resolves a revision
such as `main` from the existing cache; it does not check whether the remote branch
has changed. Prepare reuses a complete cached revision; choose a new revision
explicitly when you want another release. Loading rejects
reported missing/mismatched parameters or loader errors rather than accepting
randomly initialized replacements; no missing-key exceptions are applied.

## Text: fixed-target explanation and prompt intervention

SmolLM2-135M is a small **base** model, not a chat assistant. Prompts use the native
tokenizer with special tokens, no chat template and no truncation. Limits are
256 input tokens and 8,000 characters per prompt, with 1–64 greedy continuation
tokens. Overlong prompts are rejected rather than silently altered.

Leave the target text blank to select the original prompt's argmax next-token
ID, or provide text that tokenizes to exactly one token (leading spaces matter).
The same target ID is held fixed for the edited prompt. Each actual forward
receives graph-connected leaf `inputs_embeds`; the app differentiates that
next-token **raw logit** and sums signed gradient × embedding over dimensions.
This is not attribution for the full generated continuation, integrated
gradients, a detached-activation heuristic or an NLA description.

Side-by-side original/edited panels show readable signed token highlights,
a positive/negative evidence legend, the fixed target's logit/probability and
each prompt's continuation. Signed values remain available in the inspector. The displayed logit difference is
an input-edit intervention, not an activation patch. Tokenizations may differ;
there is no implied position alignment. Gradient sensitivity is not a causal or
semantic guarantee, and greedy decoding need not be bitwise equal across devices.

## Image: CLIP scoring and pixel IG

Upload PNG/JPEG/WebP up to 10 MiB, 4,096 pixels per axis and 16 million pixels.
EXIF orientation is applied and input converts to RGB. The native CLIP processor
resizes/crops/normalizes it; the overlay is aligned to the **model crop**, not the
uncropped upload. Supply 2–16 candidate texts (each ≤500 characters and ≤77
native tokenizer tokens) and explicitly choose the explanation target.

Scores are CLIP scaled cosine logits. Candidate-set softmax is relative to those
texts, **not calibrated class confidence**. Attribution calls AutoExplain's
existing `integrated_gradients` on normalized input pixels, with a zero normalized
baseline (channel-mean RGB) and 16 trapezoidal integration steps. It does **not**
claim Grad-ECLIP, attention attribution or concept discovery. The overlay shows
mean absolute channel attribution normalized per image, with adjustable opacity
and color map; JSON also contains signed
channel sums and the signed completeness residual. A small residual checks
integration numerics, not semantic truth or causal validity.

## Results and privacy

JSON includes model ID, a resolved commit or unverified-local label,
device/dtype, method, score convention, input text/token IDs or uploaded-image
SHA-256, target, scores and bounded-work settings. Arbitrary revision strings and
processor metadata are omitted to avoid exporting user-supplied local paths. Image exports
include attribution arrays but not original image bytes. It excludes credentials
and local checkpoint paths. **Prompt/candidate text is included**: treat exported
results as sensitive when appropriate. The app does not intentionally persist
inputs/results to disk except when you explicitly choose **Save experiment** or a
download. Models are owned by
Streamlit sessions, never a global mutable model cache. Named run history is
bounded to eight session-local records; no history is silently saved on disk.
Changing input/model/settings marks current results stale rather than presenting
them as current. No persistent pretrained attribution
hooks are installed; the real CNN's attribution hooks remain scoped.

### Explicit local experiments

Open **Save this experiment locally** beneath a Gemma/text/image/NLA result and
click **Save experiment**. **Saved experiments** provides explicit Open, JSON
export and confirmation-gated Delete. Open does not load a model or rerun the
experiment. Records survive a server restart; ordinary drafts and unsaved session
history do not. Only the selected app-owned record is deleted, never models,
caches, original reports or arbitrary paths. Deletion is not secure disk erasure.

Records live in `$XDG_DATA_HOME/autoexplain/studio/experiments.sqlite3`, falling
back to `~/.local/share/autoexplain/studio/experiments.sqlite3`. This is unencrypted
server-local storage shared by users of the same OS account/app, not a private
browser vault. Limits: 50 records, 2 MiB per JSON record, 64 MiB total payload.
Names never become paths; IDs are UUIDs and SQL values are parameterized.
Save includes input/result/model/settings provenance, but not credentials,
checkpoint paths or original image bytes. Saved image results retain the input
SHA-256 and numerical attributions; they cannot reconstruct the original image.
NLA's saved checkpoint fingerprint detects file metadata changes, not authenticated
model identity. Preserve an export if a result matters; this store is not a backup.

Task status and editable drafts survive navigation within a session. Failed or
cancelled attempts retain the previous successful Gemma/text/image output, marked
as previous when inputs change. Retry is explicit through the original action.
Native Stop remains a foreground cancellation boundary; the task is marked
cancelled only after backend cleanup unwinds. Hard process/OS termination cannot
provide that guarantee. See the [rendered UX review](studio-ux-review.md) for what
was actually observed and what remains source-reviewed only.

## Real CNN images

Select **Real CNN images**, then **Prepare pretrained CNN**. The model is
ResNet-18 with official torchvision ImageNet-1K V1 weights, not a classifier
trained on generated bars. If the approximately 45 MiB checkpoint is absent,
explicitly allow its download before Prepare. Loading checks the published
SHA-256 prefix and uses `torch.load(weights_only=True)`; there is no arbitrary
checkpoint upload, unsafe loading fallback, or random-weight inference.

The default **Example photograph · NASA astronaut** is a bundled, public-domain
NASA photograph of Eileen Collins, with source attribution and a preview. It needs
no upload or image download; predictions are not prefilled. Alternatively choose
**Upload my photograph** for a real PNG/JPEG/WebP (10 MiB, 4096 pixels per side and
16 million pixels maximum). Choose the predicted or a fixed ImageNet class and click
**Classify and explain photograph**. CPU inference produces top-five class
scores, Grad-CAM at `layer4`, and input-gradient magnitude for the same target
logit. The actual 224×224 center crop is shown beside the maps. Images and
results remain session-local; they are not sent to an external service or
persisted. Navigation away unloads the session-owned CNN; its cached weights
and last result remain. Changed inputs are labeled separately from old results.

Grad-CAM is positive, per-image-normalized evidence; absolute input gradients
show sensitivity, not signed contribution or a causal proof. ImageNet scores
are not calibrated confidence or medical validation. No weights were downloaded,
inference executed, or tests run during this replacement; runtime behavior still
needs an explicit model preparation and real photograph. The old synthetic
headless toolkit API remains available as `python -m autoexplain.demo`, but it
is no longer the Studio CNN workflow.

## NLA: activation → language → edit → controlled comparison

The NLA workspace connects the actual `Qwen/Qwen2.5-7B-Instruct` source model
with `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar`.
Explicitly prepare their artifacts (or select trusted complete local snapshots),
choose source token positions, verbalize captured block-20 activations, inspect
reconstruction diagnostics, then edit a description and compare a
residual-preserving activation patch against the original fixed target.
SmolLM is not compatible with this released pair and is never substituted.

The default is staged CPU float32: source capture, AV verbalization, AR
reconstruction/edit, then source patching, with model release between stages.
A 10 GiB RTX 3080 cannot hold this unquantized trio. Memory/disk preflight and
explicit consent precede expensive work; estimates are not guarantees and CPU
execution can be very slow. Standard, narrowly validated repository-cache blob
links can be reused without making full weight copies. See [NLA](nla.md) for
exact staged API, safety constraints, preparation controls and limitations.
Descriptions and generated outputs can confabulate; reconstruction and
same-input output changes are diagnostics, not proof of semantic faithfulness.

## Troubleshooting and boundary

- **Missing dependency/import incompatibility:** use a clean `[studio]`
  environment with Transformers 4.57.1; do not mix Transformers-5 extras.
- **Download/auth/network error:** check connectivity, model license/access and
  free cache disk. Authenticate through the Hub CLI outside the app if needed.
  Errors intentionally omit raw upstream URLs/messages that might leak secrets.
- **Local load fails:** complete explicit preparation, select the matching cached
  revision, or provide a complete compatible safetensors snapshot. No automatic
  pickle fallback, remote-code execution or missing-file download occurs.
- **OOM:** unload, reduce prompt/generation work, use a suitable CUDA dtype or CPU
  with enough RAM. There is no automatic retry or silent dtype/device change.
  Allocator cleanup cannot free memory used by other sessions/processes.
- **Slow CPU / Stop latency:** these are real forwards/gradients; image IG uses
  17 gradient evaluations plus scoring/control forwards. Stage progress yields
  to Streamlit before model operations, between generated tokens, and before
  each IG forward. Use the native top-right **Stop** control; it takes effect
  at the next yield, not in the middle of a tensor operation. A model load,
  single forward/backward, or cleanup may still take a long time. NLA also
  yields at staged boundaries and bounded generation steps. Downloads have
  owned subprocess cancellation/cleanup; inference stays in the foreground.
  No operation is silently detached or falsely marked complete by clearing UI.
- **Invalid input:** check token counts (not character counts), image bounds,
  candidate count and target-token tokenization. UI messages are deliberately
  conservative rather than displaying arbitrary upstream exception text.

There is no app authentication. Binding to all interfaces also permits non-Tailscale
connections where routing/firewall rules allow them. Restrict port 8501 with the
host firewall and Tailscale ACLs to trusted devices; do not expose it publicly.
This is not a hardened multi-user server. GitHub
Pages serves documentation only and cannot execute this workspace. No cloud
inference, general checkpoint upload, fine-tuning or affiliation claim is offered.
This revision includes sequential **Chromium rendered UX reviews** and manual
readiness/navigation/save/open interactions, documented in
[studio-ux-review.md](studio-ux-review.md). No automated tests, installations,
model downloads, training or new inference were run. Missing cache and GPU
headroom blocked live execution; runtime cancellation and completed live-model
outputs are not claimed verified.
