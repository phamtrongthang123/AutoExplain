"""Natural Language Autoencoders using the released Qwen2.5-7B L20 checkpoints.

Implements the official AV/AR checkpoint and prompt contracts through local
Hugging Face models, not an LLM-labeling callback. See docs/nla.md for provenance,
normalization, licenses, installation and the deliberately narrow support scope.
No optional imports or checkpoint loads happen at module import time.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import importlib
import json
import math
from pathlib import Path
import re

import torch

from .core import evaluating

NLA_SOURCE_REVISION = "0577769b55ad4fdd96d159e983361b97fa4e7331"
NLA_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NLA_LAYER = 20  # zero-based decoder block OUTPUT, i.e. HF hidden_states[21]
NLA_WIDTH = 3584
NLA_REPOSITORIES = {
    "source": NLA_BASE_MODEL,
    "av": "kitft/nla-qwen2.5-7b-L20-av",
    "ar": "kitft/nla-qwen2.5-7b-L20-ar",
}


def nla_memory_preflight(role, *, override=False, device="cpu", dtype=torch.float32):
    """Conservative additional working-memory estimate, not a fit guarantee.

    Includes float32 loading/workspace headroom. Call after releasing the prior
    model. Unknown or insufficient available memory requires explicit override.
    """
    if role not in NLA_REPOSITORIES:
        raise ValueError("Unknown NLA stage")
    required = (42 if role != "ar" else 34) * 1024 ** 3
    available = None
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        try:
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(fields["MemAvailable"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            pass
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        gpu_required = (34 if role != "ar" else 26) * 1024 ** 3 * (0.5 if dtype != torch.float32 else 1)
        gpu_available = torch.cuda.mem_get_info(device)[0]
        if gpu_available < gpu_required and not override:
            raise MemoryError("Insufficient conservative CUDA stage budget")
    if (available is None or available < required) and not override:
        raise MemoryError("Insufficient or unknown conservative CPU stage budget; explicit override required")
    return {"required_bytes": required, "available_bytes": available, "override": bool(override)}


def validate_nla_directory(path, role):
    """Reject arbitrary links; allow only configured, same-repository Hub blobs.

    Location validation is not cryptographic authentication. Even JSON/tokenizer
    assets are checked before optional loaders consume them.
    """
    if role not in NLA_REPOSITORIES:
        raise ValueError("Unknown NLA checkpoint role")
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Checkpoint must be a local directory")
    from huggingface_hub.constants import HF_HUB_CACHE
    repository = (Path(HF_HUB_CACHE).expanduser().resolve() /
                  ("models--" + NLA_REPOSITORIES[role].replace("/", "--")))
    known_snapshot = (root.parent == repository / "snapshots"
                      and re.fullmatch(r"[0-9a-f]{40}", root.name) is not None
                      and not repository.is_symlink()
                      and not (repository / "snapshots").is_symlink()
                      and not (repository / "blobs").is_symlink())
    for file in root.rglob("*"):
        if file.is_symlink():
            target = file.resolve(strict=True)
            if (not known_snapshot or target.parent != repository / "blobs"
                    or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target.name)
                    or not target.is_file()
                    or (repository / "blobs" / target.name).is_symlink()):
                raise ValueError("Only same-repository Hub snapshot blob links are permitted")
        elif not file.resolve(strict=True).is_relative_to(root):
            raise ValueError("Checkpoint asset escapes its directory")
    return root


def _weight_files(root, role):
    index = root / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not names:
        raise ValueError("Empty safetensors index")
    if role == "ar":
        names.add("value_head.safetensors")
    for name in names:
        if (not isinstance(name, str) or Path(name).name != name
                or not name.endswith(".safetensors") or not (root / name).is_file()):
            raise ValueError("Complete local safetensors shards are required")
    return names


@dataclass(frozen=True)
class NLAActivations:
    """One unpadded source sequence, selected rows [N,3584] in source order.

    token_ids are the *entire* source input, including chat/special tokens.
    positions indexes those IDs, not characters or AV/AR tokens. The vectors
    are raw post-residual-add block-20 outputs, before final normalization.
    Metadata is a caller assertion, not proof of the weights' provenance.
    """
    vectors: torch.Tensor
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    source_model: str = NLA_BASE_MODEL
    layer_index: int = NLA_LAYER

    def validate(self):
        _vectors(self.vectors)
        if self.source_model != NLA_BASE_MODEL or self.layer_index != NLA_LAYER:
            raise ValueError("Only Qwen2.5-7B-Instruct post-block-20 activations are supported")
        if not self.token_ids or any(type(i) is not int or i < 0 for i in self.token_ids):
            raise ValueError("token_ids must contain the full nonnegative source token sequence")
        if (len(self.positions) != len(self.vectors)
                or any(type(i) is not int or not 0 <= i < len(self.token_ids) for i in self.positions)
                or tuple(sorted(set(self.positions))) != tuple(self.positions)):
            raise ValueError("positions must be unique, increasing and aligned one-to-one with vectors")


@dataclass(frozen=True)
class NLAText:
    explanation: str
    raw_text: str
    generated_token_ids: tuple[int, ...]
    source_position: int
    source_token_id: int


@dataclass(frozen=True)
class NLAReconstruction:
    texts: tuple[NLAText, ...]
    raw_prediction: torch.Tensor
    normalized_prediction: torch.Tensor
    normalized_target: torch.Tensor
    normalized_error: torch.Tensor
    direction_mse: torch.Tensor
    cosine_similarity: torch.Tensor


def _vectors(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[0] == 0 or value.shape[1] != NLA_WIDTH
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError("Expected finite floating [N,3584] activations, N >= 1")
    norms = value.float().norm(dim=-1)
    if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Zero, near-zero or overflowing activation norms are unsupported")


def _normalize(value, scale):
    return value.float() / value.float().norm(dim=-1, keepdim=True) * scale


def nla_reconstruction_loss(prediction, target, *, mse_scale=math.sqrt(NLA_WIDTH)):
    """Differentiable per-row normalized MSE, the released AR objective component.

    Both inputs must share shape/device; this is not a GRPO trainer. At scale
    sqrt(d), the returned [N] values equal 2*(1-cosine), not raw-vector MSE.
    """
    _vectors(prediction)
    _vectors(target)
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError("Prediction and target must share shape and device")
    if not math.isfinite(mse_scale) or mse_scale <= 0:
        raise ValueError("mse_scale must be finite and positive")
    return (_normalize(prediction, mse_scale) - _normalize(target, mse_scale)).square().mean(-1)


def _optional(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError("NLA requires transformers, safetensors and PyYAML; see docs/nla.md") from exc


def _checkpoint(path, role):
    root = validate_nla_directory(path, role)
    meta = _optional("yaml").safe_load((root / "nla_meta.yaml").read_text())
    config = json.loads((root / "config.json").read_text())
    if (meta.get("kind") != "nla_model" or meta.get("schema_version") != 2
            or meta.get("role") != role or meta.get("d_model") != NLA_WIDTH
            or meta.get("extraction_layer_index") != NLA_LAYER):
        raise ValueError(f"Expected released schema-2 Qwen L20 {role} sidecar")
    if (config.get("model_type") != "qwen2" or config.get("hidden_size") != NLA_WIDTH
            or config.get("num_hidden_layers") != (28 if role == "av" else 21)
            or config.get("auto_map") or config.get("quantization_config")):
        raise ValueError("Only native, unquantized Qwen2 AV(28)/AR(21) checkpoints are supported")
    scale = meta["extraction"].get("mse_scale")
    if not isinstance(scale, (float, int)) or not math.isclose(scale, math.sqrt(NLA_WIDTH), rel_tol=1e-6):
        raise ValueError("Expected released direction-only mse_scale=sqrt(3584)")
    _weight_files(root, role)
    return root, meta


class NaturalLanguageAutoencoder:
    """Local pretrained Qwen NLA inference; two large models, one explicit device.

    Construction is inert. Legacy load() explicitly loads both checkpoints.
    Alternatively load_av()/encode(), then load_ar()/decode() keep only one
    model resident. reconstruct() still requires both via load(). No fetching.
    This native-HF execution adapter uses the released model/sidecar contracts;
    it does not call the official SGLang service or import upstream source.
    """

    def __init__(self, av_checkpoint, ar_checkpoint, *, device="cpu", dtype=torch.float32,
                 max_new_tokens=200, max_ar_tokens=512):
        self.av_checkpoint = av_checkpoint
        self.ar_checkpoint = ar_checkpoint
        self.device = torch.device(device)
        self.dtype = dtype
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only single-device CPU/CUDA inference is supported")
        if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("Expected float32, bfloat16 or float16 weights")
        if self.device.type == "cpu" and dtype != torch.float32:
            raise ValueError("Use float32 on CPU; reduced precision requires CUDA")
        if (type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512
                or type(max_ar_tokens) is not int or not 8 <= max_ar_tokens <= 4096):
            raise ValueError("Invalid generation/context budget")
        self.max_new_tokens = max_new_tokens
        self.max_ar_tokens = max_ar_tokens
        self._loaded = False
        self.av = self.ar = self.head = None
        self._progress = None
        self._cancel = None

    def _boundary(self, stage):
        if self._cancel is not None and self._cancel():
            raise InterruptedError("NLA operation cancelled")
        if self._progress is not None:
            self._progress(stage)

    def release(self):
        """Release model references; no checkpoint/cache files are removed."""
        self.av = self.ar = self.head = None
        self.av_tokenizer = self.ar_tokenizer = None
        self._loaded = False
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_av(self, *, override_memory=False, progress=None, cancel=None):
        """Release the previous stage and explicitly load only the verbalizer."""
        return self._load_stage("av", override_memory=override_memory, progress=progress, cancel=cancel)

    def load_ar(self, *, override_memory=False, progress=None, cancel=None):
        """Release the previous stage and explicitly load only the reconstructor."""
        return self._load_stage("ar", override_memory=override_memory, progress=progress, cancel=cancel)

    def _load_stage(self, role, *, override_memory, progress, cancel):
        self.release()
        self._progress, self._cancel = progress, cancel
        try:
            self._boundary("Checking memory for " + role.upper())
            nla_memory_preflight(role, override=override_memory, device=self.device, dtype=self.dtype)
            self._load_roles({role})
            self._boundary(role.upper() + " ready")
            return self
        except BaseException:
            self.release()
            raise

    def load(self):
        """Explicit local-only safetensors loading with remote code disabled."""
        if self._loaded:
            return self
        self.release()
        self._progress = self._cancel = None
        try:
            self._load_roles({"av", "ar"})
        except BaseException:
            self.release()
            raise
        return self

    def _load_roles(self, roles):
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            with torch.cuda.device(self.device):
                if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    raise ValueError("Selected CUDA device does not support bfloat16")
        av_path, av_meta = _checkpoint(self.av_checkpoint, "av")
        ar_path, ar_meta = _checkpoint(self.ar_checkpoint, "ar")
        hf = _optional("transformers")
        safe = _optional("safetensors.torch")
        options = {"local_files_only": True, "trust_remote_code": False}
        av_tokenizer = hf.AutoTokenizer.from_pretrained(str(av_path), **options)
        ar_tokenizer = hf.AutoTokenizer.from_pretrained(str(ar_path), **options)
        tokens = av_meta["tokens"]
        marker = tokens["injection_token_id"]
        if (av_tokenizer.encode(tokens["injection_char"], add_special_tokens=False) != [marker]
                or marker == av_tokenizer.unk_token_id):
            raise ValueError("AV injection marker does not match the checkpoint tokenizer")
        prompt = av_meta["prompt_templates"]["av"].format(injection_char=tokens["injection_char"])
        prompt_ids = av_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        matches = [i for i, token in enumerate(prompt_ids) if token == marker]
        if len(matches) != 1:
            raise ValueError("AV canonical prompt must contain exactly one injection marker")
        position = matches[0]
        if (not 0 < position < len(prompt_ids) - 1
                or prompt_ids[position - 1] != tokens["injection_left_neighbor_id"]
                or prompt_ids[position + 1] != tokens["injection_right_neighbor_id"]):
            raise ValueError("AV injection marker neighbors differ from the sidecar")
        injection_scale = av_meta["extraction"]["injection_scale"]
        if not isinstance(injection_scale, (int, float)) or not math.isclose(injection_scale, 150.0):
            raise ValueError("Expected released Qwen L20 injection_scale=150")
        template = ar_meta["prompt_templates"]["ar"]
        if template.count("{explanation}") != 1:
            raise ValueError("AR template must contain exactly one explanation field")
        suffix = ar_meta["tokens"]["critic_suffix_ids"]
        if not suffix:
            raise ValueError("AR checkpoint must declare its final-token suffix")
        probe = ar_tokenizer(template.format(explanation="probe"), add_special_tokens=True)["input_ids"]
        if probe[-len(suffix):] != suffix:
            raise ValueError("AR suffix/tokenizer mismatch")
        bos = ar_tokenizer.bos_token_id
        if bos is not None and ar_tokenizer("probe", add_special_tokens=True)["input_ids"][0] != bos:
            raise ValueError("AR BOS behavior differs from the training contract")
        # No device_map, adapter dispatch, pickle weights or remote implementations.
        if "av" in roles:
            self._boundary("Loading AV weights (noninterruptible loader call)")
            av, av_info = hf.AutoModelForCausalLM.from_pretrained(
                str(av_path), torch_dtype=self.dtype, use_safetensors=True,
                output_loading_info=True, **options)
            if av_info.get("missing_keys") or av_info.get("mismatched_keys") or av_info.get("error_msgs"):
                raise ValueError("Incomplete AV checkpoint; refusing randomly initialized parameters")
            self.av = av.to(self.device).eval()
            self.av.requires_grad_(False)
        if "ar" in roles:
            self._boundary("Loading AR weights (noninterruptible loader call)")
            ar, ar_info = hf.AutoModelForCausalLM.from_pretrained(
                str(ar_path), torch_dtype=self.dtype, use_safetensors=True,
                output_loading_info=True, **options)
            # Released AR omits these modules, but never backbone parameters.
            missing = set(ar_info.get("missing_keys", ())) - {"lm_head.weight", "model.norm.weight"}
            if missing or ar_info.get("mismatched_keys") or ar_info.get("error_msgs"):
                raise ValueError("Incomplete AR checkpoint; refusing randomly initialized backbone parameters")
            ar.lm_head = torch.nn.Identity()
            ar.model.norm = torch.nn.Identity()
            head = torch.nn.Linear(NLA_WIDTH, NLA_WIDTH, bias=False, dtype=self.dtype)
            head.load_state_dict(safe.load_file(str(ar_path / "value_head.safetensors")), strict=True)
            self.ar, self.head = ar.to(self.device).eval(), head.to(self.device).eval()
            self.ar.requires_grad_(False)
            self.head.requires_grad_(False)
        self.av_tokenizer, self.ar_tokenizer = av_tokenizer, ar_tokenizer
        self._prompt_ids = tuple(prompt_ids)
        self._injection_position = position
        self._injection_scale = float(injection_scale)
        self._ar_template, self._ar_suffix = template, tuple(suffix)
        self.mse_scale = float(ar_meta["extraction"]["mse_scale"])
        self._loaded = self.av is not None and self.ar is not None
        return self

    def _ready(self, role):
        models = (self.av,) if role == "av" else (self.ar, self.head)
        if any(model is None for model in models):
            raise RuntimeError("Load the required NLA stage explicitly first")
        expected = models[0].get_input_embeddings().weight.device
        for model in models:
            devices = {p.device for p in model.parameters()}
            dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
            if devices != {expected} or dtypes != {self.dtype}:
                raise ValueError("NLA weights must remain on one device with the configured dtype")
        return expected

    def encode(self, activations):
        """Verbalize each selected vector independently, greedy and bounded.

        Returns complete <explanation> contents plus raw generation and token
        provenance. Malformed/truncated generations fail, rather than pretending
        an incomplete explanation is a valid reconstruction input.
        """
        activations.validate()
        device = self._ready("av")
        ids = torch.tensor([self._prompt_ids], device=device, dtype=torch.long)
        hf = _optional("transformers")
        eos = self.av_tokenizer.eos_token_id
        if eos is None:
            raise ValueError("AV tokenizer must declare an EOS token")
        # Fresh config avoids inheriting checkpoint-specific sampling knobs.
        generation = hf.GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.0, eos_token_id=eos,
            pad_token_id=self.av_tokenizer.pad_token_id if self.av_tokenizer.pad_token_id is not None else eos,
            bos_token_id=self.av_tokenizer.bos_token_id, use_cache=True)
        owner = self

        class Boundary(hf.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                owner._boundary("AV generating (token boundary)")
                return False

        results = []
        with evaluating(self.av), torch.inference_mode():
            for vector, source_position in zip(activations.vectors, activations.positions):
                self._boundary("Verbalizing source position " + str(source_position))
                embeddings = self.av.get_input_embeddings()(ids).clone()
                embeddings[0, self._injection_position] = _normalize(
                    vector[None].to(device), self._injection_scale)[0].to(self.dtype)
                # HF inputs_embeds-only generation returns generated IDs, no
                # source/prompt ID prefix. Do not slice off len(prompt_ids).
                generated = self.av.generate(
                    inputs_embeds=embeddings, attention_mask=torch.ones_like(ids),
                    generation_config=generation,
                    stopping_criteria=hf.StoppingCriteriaList([Boundary()]))[0].detach().cpu()
                raw = self.av_tokenizer.decode(generated.tolist(), skip_special_tokens=True)
                matches = re.findall(r"<explanation>\s*(.*?)\s*</explanation>", raw, flags=re.DOTALL)
                if len(matches) != 1 or not matches[0].strip():
                    raise ValueError("AV produced no unique complete explanation; increase budget or inspect checkpoint alignment")
                results.append(NLAText(matches[0].strip(), raw, tuple(generated.tolist()),
                                       source_position, activations.token_ids[source_position]))
        return tuple(results)

    def decode(self, explanations):
        """AR text -> raw [N,3584] CPU float32 vectors, without inferred norms.

        Uses the sidecar's raw (non-chat) prompt, BOS policy and final suffix.
        These predicted magnitudes are not calibrated source activation norms.
        """
        device = self._ready("ar")
        if isinstance(explanations, str):
            explanations = [explanations]
        explanations = tuple(explanations)
        if not explanations or any(not isinstance(s, str) or not s.strip() for s in explanations):
            raise ValueError("Provide nonempty explanation strings")
        predictions = []
        with evaluating(self.ar), evaluating(self.head), torch.inference_mode():
            for text in explanations:
                self._boundary("Reconstructing explanation (one forward)")
                ids = self.ar_tokenizer(self._ar_template.format(explanation=text),
                                        add_special_tokens=True, return_tensors="pt")["input_ids"]
                if ids.shape[1] > self.max_ar_tokens:
                    raise ValueError("AR prompt exceeds max_ar_tokens; silent truncation is forbidden")
                if tuple(ids[0, -len(self._ar_suffix):].tolist()) != self._ar_suffix:
                    raise ValueError("AR final-token suffix is misaligned")
                hidden = self.ar.model(input_ids=ids.to(device), use_cache=False).last_hidden_state[0, -1]
                predictions.append(self.head(hidden).float().cpu())
                self._boundary("Explanation reconstruction complete")
        result = torch.stack(predictions)
        _vectors(result)
        return result

    def reconstruct(self, activations):
        """AV -> AR round trip with normalized error retained, not just a label."""
        texts = self.encode(activations)
        prediction = self.decode([item.explanation for item in texts])
        target = activations.vectors.detach().float().cpu()
        pred_n, target_n = _normalize(prediction, self.mse_scale), _normalize(target, self.mse_scale)
        error = target_n - pred_n
        return NLAReconstruction(texts, prediction, pred_n, target_n, error,
                                 error.square().mean(-1),
                                 torch.nn.functional.cosine_similarity(prediction, target, dim=-1))

    def edit(self, activations, original_explanations, edited_explanations, *, strength=1.0):
        """Return raw-space error-preserving decode edits, not an efficacy claim.

        x' = x + strength * ||x|| * (unit(AR(edit)) - unit(AR(original))).
        At strength=1 this retains x - ||x||*unit(AR(original)); unlike direct
        replacement it does not discard the reconstruction residual. Output is
        CPU float32 [N,D]. No hooks or source-model mutation are performed.
        """
        activations.validate()
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        before, after = self.decode(original_explanations), self.decode(edited_explanations)
        target = activations.vectors.detach().float().cpu()
        if before.shape != target.shape or after.shape != target.shape:
            raise ValueError("Need one original and edited explanation per activation")
        result = target + strength * target.norm(dim=-1, keepdim=True) * (
            _normalize(after, 1.0) - _normalize(before, 1.0))
        if not torch.isfinite(result).all():
            raise ValueError("Intervention overflow; reduce strength")
        return result


def _source(model):
    config = model.config
    if (getattr(config, "model_type", None) != "qwen2"
            or config.hidden_size != NLA_WIDTH or config.num_hidden_layers != 28):
        raise ValueError("Expected the original Qwen2.5-7B-Instruct source model")
    return model.model.layers[NLA_LAYER]


def extract_nla_activations(model, input_ids, *, positions):
    """Capture one unpadded [1,T] input at block 20, restoring modes/hooks.

    Caller owns loading/tokenization and must supply the exact source model,
    not the AV checkpoint. Architecture checks cannot prove weight identity.
    No chat formatting, padding removal, truncation or token shifting is done.
    """
    layer = _source(model)
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or input_ids.shape[1] == 0
            or input_ids.dtype != torch.long):
        raise ValueError("input_ids must be unpadded int64 [1,T]")
    if input_ids.device != model.get_input_embeddings().weight.device:
        raise ValueError("input_ids must be on the source embedding device")
    positions = tuple(positions)
    if (not positions or any(type(p) is not int or not 0 <= p < input_ids.shape[1] for p in positions)
            or tuple(sorted(set(positions))) != positions):
        raise ValueError("positions must be unique increasing source token indices")
    captured = []

    def capture(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value[0, list(positions)].detach().float().cpu().clone())

    handle = layer.register_forward_hook(capture)
    try:
        with evaluating(model), torch.inference_mode():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one source block invocation")
    result = NLAActivations(captured[0], tuple(input_ids[0].tolist()), positions)
    result.validate()
    return result


@contextmanager
def patch_nla_activations(model, activations, replacement):
    """Patch selected tokens during one full, uncached source forward only.

    Requires the same input_ids as activations.token_ids, with use_cache=False.
    Replacement is raw [N,D], same dtype/device as the block output; the caller
    explicitly converts edit()'s CPU float32 result. No broadcasting. Do not use
    for generation/KV-cache decoding or concurrently on the same model.
    """
    activations.validate()
    layer = _source(model)
    _vectors(replacement)
    if replacement.shape != activations.vectors.shape:
        raise ValueError("replacement must match the selected activation rows")
    replacement = replacement.detach().clone()
    calls = 0
    forwards = 0

    def check_input(_module, args, kwargs):
        nonlocal forwards
        forwards += 1
        ids = kwargs.get("input_ids", args[0] if args else None)
        if (forwards != 1 or len(args) > 1 or not isinstance(ids, torch.Tensor)
                or ids.dtype != torch.long
                or ids.shape != (1, len(activations.token_ids))
                or tuple(ids[0].tolist()) != activations.token_ids
                or kwargs.get("inputs_embeds") is not None
                or kwargs.get("past_key_values") is not None
                or kwargs.get("use_cache", model.config.use_cache) is not False):
            raise ValueError("NLA patch requires identical input_ids and one uncached forward (use_cache=False)")

    def patch(_module, _args, output):
        nonlocal calls
        calls += 1
        value = output[0] if isinstance(output, tuple) else output
        if calls != 1 or value.shape != (1, len(activations.token_ids), NLA_WIDTH):
            raise ValueError("NLA patch requires exactly one full matching source sequence")
        if value.device != replacement.device or value.dtype != replacement.dtype:
            raise ValueError("replacement must match source output dtype/device")
        edited = value.clone()
        edited[0, list(activations.positions)] = replacement
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    input_handle = model.register_forward_pre_hook(check_input, with_kwargs=True)
    handle = None
    try:
        handle = layer.register_forward_hook(patch)
        yield model
        if forwards != 1 or calls != 1:
            raise ValueError("NLA patch context requires exactly one forward")
    finally:
        input_handle.remove()
        if handle is not None:
            handle.remove()
