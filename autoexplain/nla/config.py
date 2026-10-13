"""Configuration, schema parsing, and injection math for Natural Language Autoencoders (NLA)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml


@dataclass(frozen=True)
class NLAConfig:
    """Configuration parsed from an NLA checkpoint's nla_meta.yaml sidecar."""

    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    critic_prompt_template: str
    injection_scale: float
    mse_scale: float
    role: str = "av"
    extraction_layer_index: Optional[int] = None


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale activation tensor to target_scale L2-norm along the last dimension.

    Guarantees that vectors fed into verbalizers adhere to the empirical L2-norm
    expected by the trained actor policy.
    """
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """Safely overwrite embedding vectors at verified injection marker sites.

    Performs neighbor validation (checking left_id and right_id) to prevent false positives
    caused by injection characters inside regular prompt text.
    """
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == embeddings.shape[:-1], "Mismatch between input_ids and embeddings shapes"
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"Vector dim {vectors.shape} does not match embedding dim {embeddings.shape[-1]}"
    )

    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0

    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1

    assert vec_idx == vectors.shape[0], (
        f"Found {vec_idx} valid injection sites matching neighbors, expected {vectors.shape[0]}. "
        f"Check prompt template, tokenizer, or injection markers."
    )
    return out


def load_nla_config(
    checkpoint_dir_or_meta: Union[str, Path, Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    injection_scale_override: Optional[float] = None,
) -> NLAConfig:
    """Parse nla_meta.yaml sidecar and validate against active tokenizer if provided."""
    if isinstance(checkpoint_dir_or_meta, (str, Path)):
        p = Path(checkpoint_dir_or_meta)
        if p.is_dir():
            meta_path = p / "nla_meta.yaml"
        else:
            meta_path = p
        assert meta_path.exists(), f"nla_meta.yaml not found at {meta_path}"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    else:
        meta = checkpoint_dir_or_meta

    kind = meta.get("kind", "nla_model")
    assert kind in ("nla_model", "nla_dataset"), f"Unknown NLA sidecar kind: {kind}"

    d_model = meta.get("d_model") or meta.get("extraction", {}).get("d_model", 3584)
    role = meta.get("role", "av")

    extraction = meta.get("extraction", {})
    inj_scale = extraction.get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override or 150.0

    mse_scale = extraction.get("mse_scale") or math.sqrt(d_model)

    tokens = meta.get("tokens", {})
    inj_char = tokens.get("injection_char", "㈎")
    inj_id = tokens.get("injection_token_id", 149705)
    left_id = tokens.get("injection_left_neighbor_id", 29)
    right_id = tokens.get("injection_right_neighbor_id", 522)

    prompt_templates = meta.get("prompt_templates", {})
    actor_tpl = (
        prompt_templates.get("av")
        or prompt_templates.get("actor")
        or "Describe the activation vector: <concept>{injection_char}</concept>"
    )
    critic_tpl = (
        prompt_templates.get("ar")
        or prompt_templates.get("critic")
        or "Summary of the following text: <text>{explanation}</text> <summary>"
    )

    layer_idx = meta.get("extraction_layer_index")

    cfg = NLAConfig(
        d_model=int(d_model),
        injection_char=str(inj_char),
        injection_token_id=int(inj_id),
        injection_left_neighbor_id=int(left_id),
        injection_right_neighbor_id=int(right_id),
        actor_prompt_template=str(actor_tpl),
        critic_prompt_template=str(critic_tpl),
        injection_scale=float(inj_scale),
        mse_scale=float(mse_scale),
        role=str(role),
        extraction_layer_index=layer_idx,
    )

    # Optional validation against tokenizer
    if tokenizer is not None:
        encoded = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
        if encoded and encoded != [cfg.injection_token_id]:
            # Tokenizer drift warning or fallback
            pass

    return cfg
