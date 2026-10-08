"""Bounded, local-only execution for the curated AutoExplain studio.

No model is downloaded or loaded at import. LoadedModel belongs to one UI session.
"""
from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import json
import shutil

import torch

LIBRARY = {
    "HuggingFaceTB/SmolLM2-135M": {
        "kind": "text", "model_type": "llama", "parameters": 135_000_000,
        "disk_gb": 0.6, "runtime_gb": "1–3 GB CPU RAM / 1–2 GB CUDA VRAM",
        "description": "Small base causal LM (not an instruction/chat model).",
    },
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
        "kind": "image", "model_type": "clip", "parameters": 151_000_000,
        "disk_gb": 0.7, "runtime_gb": "2–4 GB CPU RAM / 1–3 GB CUDA VRAM",
        "description": "Native Transformers CLIP; image/candidate-text similarity.",
    },
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resources():
    result = {"home_disk_free_gb": round(shutil.disk_usage(Path.home()).free / 1e9, 2)}
    try:
        import psutil
        result["available_ram_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    except ImportError:
        result["available_ram_gb"] = "Install psutil for live RAM information"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        result.update(cuda0_free_gb=round(free / 1e9, 2), cuda0_total_gb=round(total / 1e9, 2))
    return result


def recommend(model_id):
    """Conservative live defaults; estimates are not allocation guarantees."""
    if model_id not in LIBRARY:
        raise ValueError("Unsupported model")
    info = resources()
    cuda = isinstance(info.get("cuda0_free_gb"), (int, float)) and info["cuda0_free_gb"] >= 3
    return {"device": "cuda:0" if cuda else "cpu", "dtype": "float32",
            "resources": info,
            "guidance": ("CUDA has at least 3 GB free; float32 preserves attribution precision. " if cuda else
                         "CPU float32 recommended; reserve approximately 4 GB RAM. ") +
                        "These small-model estimates include rough workspace headroom, not measured peaks or guarantees. "
                        "Other sessions, gradients and load buffers consume memory. NLA has separate, much larger CPU requirements."}


def _complete_snapshot(directory, kind):
    """Check required local assets without loading weights or touching the network."""
    directory = Path(directory)
    def present(name):
        path = directory / name
        return path.is_file() and path.stat().st_size > 0
    if not present("config.json") or not present("tokenizer_config.json"):
        return False
    if not (present("tokenizer.json") or (present("vocab.json") and present("merges.txt"))):
        return False
    if kind == "image" and not present("preprocessor_config.json"):
        return False
    if present("model.safetensors.index.json"):
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            return False
        shards = list(index["weight_map"].values())
        return bool(shards) and all(isinstance(name, str) and Path(name).name == name
                                    and name.endswith(".safetensors") and present(name) for name in shards)
    return present("model.safetensors")


def readiness(model_id, revision="main", path=""):
    if model_id not in LIBRARY:
        raise ValueError("Unsupported model")
    try:
        directory = Path(path).expanduser() if path.strip() else Path(cached_path(model_id, revision))
        complete = _complete_snapshot(directory, LIBRARY[model_id]["kind"])
        return {"downloaded": complete, "message": "Required local assets found; not yet loaded or weight-validated." if complete else
                "Local snapshot is incomplete. Prepare with download consent, or select a complete trusted local directory."}
    except (OSError, ValueError, KeyError):
        return {"downloaded": False, "message": "No complete local snapshot found for this revision. Prepare requires explicit download consent."}


def _stage(progress, stage, current=0, total=1):
    if progress is not None:
        progress(stage, current, total)


def prepare_model(model_id, *, revision="main", path="", device="cpu", dtype="float32", consent=False, progress=None):
    """Reuse a complete snapshot; only explicit consent permits missing-asset fetches."""
    _stage(progress, "Checking local assets")
    ready = readiness(model_id, revision, path)
    if not ready["downloaded"]:
        if path.strip():
            raise ValueError("Complete the trusted local directory or clear the path to use Hub preparation")
        if not consent:
            raise ValueError("Explicit download consent required")
        download_model(model_id, revision, progress=lambda elapsed, message:
                       _stage(progress, f"Downloading missing assets — {elapsed}s; {message}"))
        if not readiness(model_id, revision)["downloaded"]:
            raise ValueError("Downloaded snapshot lacks required assets")
    _stage(progress, "Loading local safetensors (Stop applies after this load)")
    loaded = load_model(model_id, path=path, revision=revision, device=device, dtype=dtype)
    try:
        _stage(progress, "Ready to explain", 1, 1)
    except BaseException:
        unload(loaded)
        raise
    return loaded


def safe_error(exc):
    """Do not echo upstream URLs, tokens, local paths or arbitrary exception text."""
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in str(exc).lower():
        return "Out of memory. Unload the model, reduce input/work budget, or select CPU/another dtype. No automatic retry was made."
    if isinstance(exc, ImportError):
        return "A local dependency is missing or incompatible. Install the studio extra in a clean environment (Transformers 4.57.1); do not combine with Transformers-5 extras."
    actionable = {
        "Prompt must contain 1–8000 characters", "Prompt must contain 1–256 tokens",
        "Target must tokenize to exactly one token", "Candidate exceeds 77 tokens",
        "Source must contain 1–128 tokens", "Target must be exactly one token",
        "Empty source text", "Edited explanations cannot be empty",
        "Checkpoint assets changed; validate again", "CPU requires float32",
        "CUDA unavailable", "bfloat16 unavailable", "Image dimensions too large",
        "Nonfinite explanation; try float32",
    }
    if isinstance(exc, ValueError) and str(exc) in actionable:
        return str(exc) + ". Correct this setting and retry; no automatic fallback was made."
    if isinstance(exc, ValueError):
        return "Input or checkpoint validation failed. Check the documented model, safetensors snapshot, dimensions, token and device/dtype limits."
    return "Operation failed. Check available memory/disk, complete safetensors files and compatible dependencies. For downloads, check connectivity and model access/license; authenticate with the Hub CLI outside the app if needed. Error details are intentionally not displayed to avoid leaking credentials."


def download_model(model_id, revision="main", progress=None):
    """Explicit foreground Hub snapshot download; incomplete blobs resume on retry."""
    if model_id not in LIBRARY:
        raise ValueError("Unsupported model")
    import subprocess
    import sys
    import time
    import tempfile
    # Owned subprocess, never detached: the UI remains interruptible even during
    # a large blob transfer. Streamlit Stop raises from progress(); finally kills
    # the child and waits for cleanup. No inference/GPU work runs here.
    script = """
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm
class Progress(tqdm):
    def update(self, n=1):
        result = super().update(n)
        print(f'{int(self.n)}/{int(self.total or 0)} files processed', flush=True)
        return result
snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=1, tqdm_class=Progress,
    allow_patterns=['*.json', 'model.safetensors', 'model-*.safetensors', '*.txt', '*.model', '*.jinja'])
"""
    # A temporary progress stream avoids pipe backpressure and includes no model
    # inputs or credentials. It is removed on success, failure and cancellation.
    with tempfile.TemporaryFile(mode="w+b") as updates:
        child = subprocess.Popen([sys.executable, "-c", script, model_id, revision],
                                 stdout=updates, stderr=subprocess.DEVNULL)
        started, position, message = time.monotonic(), 0, "Resolving files / transferring"
        try:
            while child.poll() is None:
                # pread does not change the child's shared file offset.
                import os
                chunk = os.pread(updates.fileno(), 65536, position) if hasattr(os, "pread") else b""
                if chunk:
                    position += len(chunk)
                    message = chunk.decode("utf-8", errors="replace").splitlines()[-1]
                if progress is not None:
                    progress(int(time.monotonic() - started), message)
                time.sleep(0.25)
            if child.returncode:
                raise RuntimeError("Hub download failed")
            return cached_path(model_id, revision)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def cached_path(model_id, revision="main"):
    if model_id not in LIBRARY:
        raise ValueError("Unsupported model")
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision, local_files_only=True)


@dataclass
class LoadedModel:
    model: object
    processor: object
    metadata: dict


def load_model(model_id, *, path="", revision="main", device="cpu", dtype="float32"):
    if model_id not in LIBRARY or device not in ("cpu", "cuda:0"):
        raise ValueError("Unsupported model/device")
    if dtype not in ("float32", "float16", "bfloat16") or (device == "cpu" and dtype != "float32"):
        raise ValueError("CPU requires float32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 unavailable")
    from transformers import AutoTokenizer, CLIPProcessor, CLIPModel, LlamaForCausalLM
    directory = Path(path).expanduser().resolve() if path.strip() else Path(cached_path(model_id, revision)).resolve()
    config = json.loads((directory / "config.json").read_text())
    spec = LIBRARY[model_id]
    if config.get("model_type") != spec["model_type"] or config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("Unsupported configuration")
    # Constrain architecture sizes as well as class. Local weights are not authenticated.
    if spec["kind"] == "text":
        if (config.get("hidden_size"), config.get("num_hidden_layers"), config.get("vocab_size"), config.get("intermediate_size"), config.get("num_attention_heads"), config.get("num_key_value_heads")) != (576, 30, 49152, 1536, 9, 3):
            raise ValueError("Not the supported SmolLM2 architecture")
    else:
        vision, text = config.get("vision_config", {}), config.get("text_config", {})
        if (vision.get("hidden_size"), vision.get("num_hidden_layers"), vision.get("image_size"), vision.get("patch_size"), text.get("hidden_size"), text.get("num_hidden_layers")) != (768, 12, 224, 32, 512, 12):
            raise ValueError("Not the supported CLIP architecture")
        if (vision.get("intermediate_size"), text.get("intermediate_size"), text.get("vocab_size"), config.get("projection_dim")) != (3072, 2048, 49408, 512):
            raise ValueError("Unsupported CLIP dimensions")
    if not list(directory.glob("*.safetensors")):
        raise ValueError("Safetensors required")
    local = dict(local_files_only=True, trust_remote_code=False)
    processor = (AutoTokenizer if spec["kind"] == "text" else CLIPProcessor).from_pretrained(directory, **local)
    cls = LlamaForCausalLM if spec["kind"] == "text" else CLIPModel
    model, loading_info = cls.from_pretrained(
        directory, use_safetensors=True, torch_dtype=getattr(torch, dtype),
        output_loading_info=True, **local,
    )
    # Native loaders account for their tied weights. Do not exempt missing keys:
    # a successful constructor must not conceal randomly initialized parameters.
    if any(loading_info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        del model
        raise ValueError("Incomplete or incompatible checkpoint parameters")
    model = model.to(device).eval()
    model.requires_grad_(False)
    from huggingface_hub.constants import HF_HUB_CACHE
    import re
    cache_repository = (Path(HF_HUB_CACHE).expanduser() / ("models--" + model_id.replace("/", "--"))).resolve()
    known_cache_snapshot = (
        directory.parent == cache_repository / "snapshots"
        and re.fullmatch(r"[0-9a-f]{40}", directory.name) is not None
    )
    commit = directory.name if known_cache_snapshot else "unverified local snapshot"
    return LoadedModel(model, processor, {
        "model_id": model_id, "kind": spec["kind"], "requested_revision": revision,
        "resolved_revision": commit, "config_sha256": fingerprint(config),
        "local_custom_path": bool(path.strip()), "device": device, "dtype": dtype,
        "snapshot_provenance": "configured Hub cache / selected repository / commit directory" if known_cache_snapshot else "unverified local directory",
        "provenance": "Cache location and commit name are metadata only, not weight authentication. Local files are not cryptographically authenticated as the selected release.",
    })


def unload(loaded):
    if loaded is not None:
        loaded.model = None
        loaded.processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def explain_text(loaded, prompt, *, comparison="", target_text="", max_new_tokens=32, progress=None):
    if loaded.metadata["kind"] != "text" or not 1 <= max_new_tokens <= 64:
        raise ValueError("Unsupported execution")
    tokenizer, model = loaded.processor, loaded.model
    device = loaded.metadata["device"]

    def encode(text):
        if not text.strip() or len(text) > 8000:
            raise ValueError("Prompt must contain 1–8000 characters")
        batch = tokenizer(text, return_tensors="pt", truncation=False)
        if not 1 <= batch["input_ids"].shape[1] <= 256:
            raise ValueError("Prompt must contain 1–256 tokens")
        return {key: value.to(device) for key, value in batch.items()}

    _stage(progress, "Validating text inputs")
    original = encode(prompt)
    other = encode(comparison) if comparison.strip() else None
    _stage(progress, "Selecting original fixed next-token target")
    with torch.no_grad():
        logits = model(**original, use_cache=False).logits[0, -1]
        target = int(logits.argmax())
    if target_text:
        ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("Target must tokenize to exactly one token")
        target = ids[0]

    def run(text, batch, label):
        from transformers import StoppingCriteria, StoppingCriteriaList
        class YieldProgress(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                _stage(progress, f"{label}: generating", input_ids.shape[1] - batch["input_ids"].shape[1], max_new_tokens)
                return False
        _stage(progress, f"{label}: generating", 0, max_new_tokens)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                       num_beams=1, pad_token_id=tokenizer.eos_token_id,
                                       stopping_criteria=StoppingCriteriaList([YieldProgress()]))
        _stage(progress, f"{label}: signed gradient attribution (one forward/backward)")
        # These leaf embeddings are the actual model inputs, not detached captures
        # of a separate forward. Parameters need not require gradients.
        with torch.enable_grad():
            embeddings = model.get_input_embeddings()(batch["input_ids"]).detach().requires_grad_(True)
            scores = model(inputs_embeds=embeddings, attention_mask=batch["attention_mask"], use_cache=False).logits[0, -1].float()
            gradient = torch.autograd.grad(scores[target], embeddings)[0]
            if not torch.isfinite(scores).all() or not torch.isfinite(gradient).all():
                raise ValueError("Nonfinite explanation; try float32")
            attribution = (gradient.float() * embeddings.float()).sum(-1)[0].detach().cpu().tolist()
        return {"prompt": text, "input_ids": batch["input_ids"][0].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist()),
                "attribution": attribution, "target_logit": float(scores[target].detach()),
                "target_probability": float(scores.detach().softmax(-1)[target]),
                "continuation": tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)}

    rows = [run(prompt, original, "Original")]
    if other is not None:
        rows.append(run(comparison, other, "Edited"))
    _stage(progress, "Explanation complete", 1, 1)
    return {"model": loaded.metadata.copy(), "method": "signed input-embedding gradient × input, summed over embedding dimensions",
            "score": "fixed next-token raw logit (same token in both prompts)",
            "target_id": target, "target_token": tokenizer.decode([target]), "results": rows,
            "settings": {"max_new_tokens": max_new_tokens, "do_sample": False, "max_input_tokens": 256},
            "preprocessing": "native tokenizer; special tokens enabled; no chat template; no truncation"}


def explain_image(loaded, image_bytes, candidates, target, progress=None):
    if loaded.metadata["kind"] != "image" or not 2 <= len(candidates) <= 16 or not 0 <= target < len(candidates):
        raise ValueError("Unsupported image execution")
    if len(image_bytes) > 10 * 1024 * 1024 or any(not text.strip() or len(text) > 500 for text in candidates):
        raise ValueError("Upload/text too large")
    from io import BytesIO
    from PIL import Image, ImageOps
    from .attribution import integrated_gradients
    _stage(progress, "Validating and preprocessing image")
    image = Image.open(BytesIO(image_bytes))
    if image.width > 4096 or image.height > 4096 or image.width * image.height > 16_000_000:
        raise ValueError("Image dimensions too large")
    image = ImageOps.exif_transpose(image).convert("RGB")
    model, processor = loaded.model, loaded.processor
    # Reject long captions rather than silently changing the candidate strings.
    if any(len(ids) > 77 for ids in processor.tokenizer(candidates, truncation=False)["input_ids"]):
        raise ValueError("Candidate exceeds 77 tokens")
    batch = processor(text=candidates, images=image, return_tensors="pt", padding=True, truncation=False)
    batch = {key: value.to(loaded.metadata["device"]) for key, value in batch.items()}
    pixels = batch.pop("pixel_values").to(dtype=next(model.parameters()).dtype)

    class Scores(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = model
            self.evaluations = 0

        def forward(self, value):
            # 1 initial score + 17 IG samples + 2 completeness control forwards.
            _stage(progress, "Image scoring / integrated gradients", self.evaluations, 20)
            result = self.clip(pixel_values=value, **batch).logits_per_image
            self.evaluations += 1
            return result

    adapter = Scores()
    with torch.no_grad():
        logits = adapter(pixels)[0].float().cpu()
    result = integrated_gradients(adapter, pixels, target, baseline=0.0, steps=16)
    if not torch.isfinite(logits).all() or not torch.isfinite(result.attributions).all() or not torch.isfinite(result.completeness_delta).all():
        raise ValueError("Nonfinite explanation; try float32")
    signed = result.attributions[0].float().sum(0).cpu()
    magnitude = result.attributions[0].float().abs().mean(0).cpu()
    normalized = magnitude / magnitude.max().clamp_min(1e-12)
    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean)[:, None, None]
    std = torch.tensor(ip.image_std)[:, None, None]
    display = (pixels[0].detach().float().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    record = {"model": loaded.metadata.copy(), "method": "integrated gradients on normalized image pixels",
              "score": "CLIP scaled cosine logit for fixed candidate", "target_index": target,
              "candidates": candidates, "logits": logits.tolist(), "candidate_softmax": logits.softmax(-1).tolist(),
              "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
              "preprocessing": ip.to_dict(), "settings": {"steps": 16, "baseline": "zero normalized pixels (channel mean RGB)", "text_max_tokens": 77},
              "completeness_residual": float(result.completeness_delta.item()),
              "signed_channel_sum": signed.tolist(), "mean_absolute_channel_attribution": magnitude.tolist()}
    _stage(progress, "Image explanation complete", 20, 20)
    return record, display, normalized.numpy()
