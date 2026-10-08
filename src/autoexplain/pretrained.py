"""Pinned real Gemma 4 experiments with explicit resource and activation contracts.

This module does not accept model licenses, execute remote model code, or silently
fall back to a random model. Model files live in the user's Hugging Face cache.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import torch
from torch import nn
from .core import evaluating

GEMMA4_ID = "google/gemma-4-E2B"
GEMMA4_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
GEMMA4_FILES = ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "model.safetensors"]


class CPUEmbeddingBridge(nn.Module):
    """Keep an embedding table on CPU; copy only selected rows to the input device.

    Unlike weight offloading, the entire table is never temporarily moved to GPU.
    CPU and CUDA embedding scaling may have small precision differences. The
    bridge preserves autograd through its output but is not a training/offload
    framework. Do not call model.to(cuda) after installing this bridge.
    """
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding.cpu()

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        return self.embedding(input_ids.to("cpu")).to(input_ids.device)


def place_gemma4(model, *, decoder_device="cuda:0"):
    """Mutate a dedicated Gemma4ForCausalLM: CPU PLE table, requested decoder device."""
    from transformers import Gemma4ForCausalLM
    if not isinstance(model, Gemma4ForCausalLM):
        raise TypeError("Expected native Gemma4ForCausalLM, not an arbitrary architecture")
    if isinstance(model.model.embed_tokens_per_layer, CPUEmbeddingBridge):
        raise ValueError("Model is already split; construct a fresh model")
    embedding = model.model.embed_tokens_per_layer
    # Remove the table temporarily so model.to() cannot copy its 4.7GB to GPU.
    model.model.embed_tokens_per_layer = nn.Identity()
    try:
        model.to(decoder_device)
    finally:
        model.model.embed_tokens_per_layer = CPUEmbeddingBridge(embedding)
    model.hf_device_map = {"model.embed_tokens_per_layer": "cpu", "": str(decoder_device)}
    return model


@dataclass
class PretrainedBundle:
    model: nn.Module
    tokenizer: object
    metadata: dict


def load_gemma4(*, allow_download=False, decoder_device="cuda:0", dtype=torch.bfloat16,
                min_host_available_gib=18, min_gpu_free_gib=6):
    """Load pinned official E2B TEXT weights. Default is cache-only and BF16.

    ~10.25GB official multimodal checkpoint is cached once; only the text model
    is instantiated. The 4.70GB PLE table stays on CPU. No quantization or remote
    code. CPU mode is available but throughput is not promised. The dedicated
    GPU environment must have matched torch/vision/audio CUDA builds.
    """
    import psutil
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig
    device = torch.device(decoder_device)
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Only explicitly validated BF16/FP32 loading modes are offered")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA decoder placement is supported")
    if psutil.virtual_memory().available < min_host_available_gib * 1024**3:
        raise RuntimeError("Insufficient currently available host memory for this loading budget")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required; use the dedicated LLM environment")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This device does not advertise BF16 support")
        if torch.cuda.mem_get_info(device)[0] < min_gpu_free_gib * 1024**3:
            raise RuntimeError("Insufficient currently free GPU memory; no automatic eviction")
    started = time.perf_counter()
    snapshot = snapshot_download(GEMMA4_ID, revision=GEMMA4_REVISION,
                                 allow_patterns=GEMMA4_FILES, local_files_only=not allow_download,
                                 max_workers=2)
    config = Gemma4TextConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    from transformers.utils import logging as hf_logging
    previous_verbosity = hf_logging.get_verbosity()
    previous_progress = hf_logging.is_progress_bar_enabled()
    try:
        # Native text loading intentionally ignores the checkpoint's audio/vision
        # weights. Keep the notebook readable, but inspect loading_info below.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        model, loading = Gemma4ForCausalLM.from_pretrained(
            snapshot, config=config, dtype=dtype, device_map="cpu", local_files_only=True,
            trust_remote_code=False, attn_implementation="eager", output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."})
    finally:
        hf_logging.set_verbosity(previous_verbosity)
        if previous_progress:
            hf_logging.enable_progress_bar()
    # Never accept randomly initialized missing text parameters as a real checkpoint.
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Pretrained text weights did not load completely; refusing random fallback")
    permitted_nontext = ("model.audio_tower.", "model.vision_tower.", "model.embed_audio.", "model.embed_vision.")
    if any(not key.startswith(permitted_nontext) for key in loading.get("unexpected_keys", [])):
        raise RuntimeError("Unexpected non-media checkpoint weights; review mapping before inference")
    model.eval().requires_grad_(False)
    if device.type == "cuda":
        ple = model.model.embed_tokens_per_layer.weight
        decoder_bytes = sum(p.numel()*p.element_size() for p in model.parameters()) - ple.numel()*ple.element_size()
        if torch.cuda.mem_get_info(device)[0] < decoder_bytes + 1024**3:
            raise RuntimeError("Decoder weights plus 1GiB headroom exceed currently free VRAM")
    place_gemma4(model, decoder_device=str(device))
    metadata = {"model_id": GEMMA4_ID, "revision": GEMMA4_REVISION,
                "dtype": str(dtype), "quantized": False,
                "decoder_device": str(device), "per_layer_embedding_device": "cpu",
                "text_parameters": sum(p.numel() for p in model.parameters()),
                "cpu_ple_parameters": model.model.embed_tokens_per_layer.weight.numel(),
                "load_seconds": time.perf_counter()-started,
                "ignored_nontext_weight_count": len(loading.get("unexpected_keys", [])),
                "missing_text_weights": 0}
    return PretrainedBundle(model, tokenizer, metadata)


def encode_prompt(bundle, text, *, max_tokens=32):
    """Single unpadded prompt; refuse truncation that could change the question."""
    ids = bundle.tokenizer.encode(text, add_special_tokens=True)
    if not 2 <= len(ids) <= max_tokens:
        raise ValueError(f"Prompt has {len(ids)} tokens; require 2–{max_tokens}, no silent truncation")
    return torch.tensor([ids], device=bundle.model.get_input_embeddings().weight.device)


def single_token_target(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("This experiment requires a single-token completion; choose another target")
    return ids[0]


@torch.no_grad()
def next_logits(model, token_ids):
    with evaluating(model):
        logits = model(input_ids=token_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    if not logits.isfinite().all():
        raise ValueError("Nonfinite next-token logits")
    return logits


def margin(logits, target, foil):
    if target == foil:
        raise ValueError("Target and foil must differ")
    return float((logits[target]-logits[foil]).detach())


@contextmanager
def record_residuals(model, layers):
    """Detached post-block residuals for Gemma4; one call per selected layer."""
    if not layers or len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(model.model.layers) for i in layers):
        raise ValueError("Supply unique valid decoder layer indices")
    records, handles = {}, []
    def capture(index):
        def hook(module, args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError("Expected Gemma4 post-block tensor [B,T,D]")
            if index in records:
                raise ValueError("Layer ran twice; record a single forward with no cache")
            records[index] = output.detach().clone()
        return hook
    try:
        for index in layers:
            handles.append(model.model.layers[index].register_forward_hook(capture(index)))
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_residual(model, layer, donor, *, positions):
    """Patch specific token positions at one post-block residual; no KV manipulation."""
    if not 0 <= layer < len(model.model.layers) or not positions:
        raise ValueError("Invalid layer/positions")
    if donor.ndim != 3 or any(type(p) is not int or not 0 <= p < donor.shape[1] for p in positions):
        raise ValueError("Invalid donor or token positions")
    value = donor.detach().clone()
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.shape != value.shape or output.device != value.device or output.dtype != value.dtype:
            raise ValueError("Donor and recipient activations must match shape/device/dtype")
        changed = output.clone()
        changed[:, positions] = value[:, positions]
        return changed
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


@torch.no_grad()
def layer_readouts(model, residuals, target, foil):
    """Plain logit-lens readout with the model's real final norm, head AND softcap."""
    rows = []
    for layer, h in residuals.items():
        logits = model.lm_head(model.model.norm(h[:, -1:]))[0, -1]
        cap = model.config.final_logit_softcapping
        if cap is not None:
            logits = (logits/cap).tanh()*cap
        logits = logits.float()
        rows.append({"layer": layer, "margin": margin(logits,target,foil),
                     "top_token_id": int(logits.argmax())})
    return rows


def embedding_attribution(model, token_ids, target, foil):
    """Gradient×activation for BOTH main and per-layer embedding routes.

    Gemma4 has token identity entering every layer. Replacing inputs_embeds alone
    would miss that path and can trigger embedding reversal. We keep input_ids
    intact and differentiate separate hooked outputs. Scores are signed route-
    conditional sensitivities, not an additive completeness claim across routes.
    """
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Use a dedicated frozen model to avoid parameter-gradient storage")
    captured, handles = {}, []
    modules = {"main_embedding": model.model.embed_tokens,
               "per_layer_embedding": model.model.embed_tokens_per_layer}
    def hook_for(name):
        def hook(module, args, output):
            if name in captured:
                raise ValueError("Embedding site executed more than once")
            value = output.detach().clone().requires_grad_(True)
            captured[name] = value
            return value
        return hook
    try:
        with evaluating(model), torch.enable_grad():
            for name, module in modules.items():
                handles.append(module.register_forward_hook(hook_for(name)))
            logits = model(input_ids=token_ids, use_cache=False, logits_to_keep=1).logits[0,-1].float()
            score = logits[target]-logits[foil]
            gradients = torch.autograd.grad(score, list(captured.values()))
            result = {}
            for (name, activation), gradient in zip(captured.items(), gradients):
                if not gradient.isfinite().all():
                    raise ValueError("Nonfinite embedding gradient")
                result[name] = {"gradient_l2": gradient.detach().float().norm(dim=-1)[0].cpu().tolist(),
                                "gradient_x_activation": (gradient.float()*activation.float()).sum(-1)[0].detach().cpu().tolist()}
            return result
    finally:
        for handle in handles:
            handle.remove()


def random_matched_donor(recipient, donor, *, positions, seed):
    """Random direction matched to donor-minus-recipient norm at each patched token."""
    generator = torch.Generator(device=recipient.device).manual_seed(seed)
    result = recipient.detach().clone()
    reference = (donor[:,positions]-recipient[:,positions]).float()
    noise = torch.randn(reference.shape, device=reference.device, generator=generator)
    noise = noise / noise.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    result[:,positions] += (noise*reference.norm(dim=-1,keepdim=True)).to(result.dtype)
    return result
