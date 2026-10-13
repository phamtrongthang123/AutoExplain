"""Activation Verbalizer (AV / Actor) for Natural Language Autoencoders."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from autoexplain.nla.config import (
    NLAConfig,
    inject_at_marked_positions,
    load_nla_config,
    normalize_activation,
)


class ActivationVerbalizer:
    """Actor module translating neural activation vectors into natural-language descriptions.

    Supports both in-process PyTorch execution (via Hugging Face `inputs_embeds`)
    and remote SGLang rollout servers.
    """

    def __init__(
        self,
        checkpoint_dir_or_meta: Optional[Union[str, Path, Dict[str, Any]]] = None,
        config: Optional[NLAConfig] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        sglang_url: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sglang_url = sglang_url

        if config is not None:
            self.cfg = config
        elif checkpoint_dir_or_meta is not None:
            self.cfg = load_nla_config(checkpoint_dir_or_meta)
        else:
            # Default fallback configuration for general models (e.g. GPT-2 or custom dims)
            self.cfg = NLAConfig(
                d_model=768,
                injection_char="㈎",
                injection_token_id=149705,
                injection_left_neighbor_id=29,
                injection_right_neighbor_id=522,
                actor_prompt_template="Explain activation: <concept>{injection_char}</concept>",
                critic_prompt_template="Summary of text: <text>{explanation}</text> <summary>",
                injection_scale=150.0,
                mse_scale=math.sqrt(768),
            )

        self.tokenizer = tokenizer
        self.model = model

        # If a checkpoint path was passed and model/tokenizer not yet loaded
        if checkpoint_dir_or_meta and isinstance(checkpoint_dir_or_meta, (str, Path)) and not model and not sglang_url:
            p = Path(checkpoint_dir_or_meta)
            if p.exists() and (p / "config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(p),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(self.device).eval()

    def extract_explanation_tags(self, text: str) -> str:
        """Extract content inside <explanation> tags if present; otherwise sanitize and return."""
        pattern = r"<explanation>(.*?)</explanation>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: strip prompt artifacts if present
        cleaned = re.sub(r"<concept>.*?</concept>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def verbalize(
        self,
        activation_vector: torch.Tensor,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Convert activation vector into natural language explanation.

        Args:
            activation_vector: Tensor of shape (d_model,) or (1, d_model)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary containing 'explanation', 'raw_text', and 'injection_status'.
        """
        vec = activation_vector.detach().float().view(1, -1)
        if vec.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"Activation vector dimension {vec.shape[-1]} does not match verbalizer d_model {self.cfg.d_model}"
            )

        # Rescale vector to match trained actor distribution
        v_scaled = normalize_activation(vec, self.cfg.injection_scale)

        # Path 1: Remote SGLang client
        if self.sglang_url:
            return self._verbalize_sglang(v_scaled, max_new_tokens, temperature)

        # Path 2: In-process Hugging Face model
        if self.model is not None and self.tokenizer is not None:
            return self._verbalize_hf(v_scaled, max_new_tokens, temperature)

        # Path 3: Lightweight deterministic concept projection (standalone/mock mode)
        return self._verbalize_heuristic(v_scaled)

    def _verbalize_hf(
        self,
        v_scaled: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """In-process generation using Hugging Face model.generate with inputs_embeds."""
        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_ids = self.tokenizer.encode(content, add_special_tokens=True)
        else:
            prompt_ids = self.tokenizer.encode(content, add_special_tokens=True)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
        # Get base word embeddings
        embed_layer = self.model.get_input_embeddings()
        embeds = embed_layer(input_ids).float()

        # Overwrite injection slot
        seq_len = input_ids.shape[-1]
        injected = False
        for p in range(seq_len):
            if input_ids[0, p].item() == self.cfg.injection_token_id:
                embeds[0, p] = v_scaled.to(embeds.device, embeds.dtype)
                injected = True
                break

        if not injected:
            # Fallback: inject at middle token position if exact ID not matched
            mid = seq_len // 2
            embeds[0, mid] = v_scaled.to(embeds.device, embeds.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs_embeds=embeds.to(self.model.dtype),
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = self.extract_explanation_tags(raw_text)

        return {
            "explanation": explanation,
            "raw_text": raw_text,
            "injection_status": "hf_inputs_embeds_success",
        }

    def _verbalize_sglang(
        self,
        v_scaled: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Delegate generation to external SGLang server."""
        import httpx

        # Prepare HTTP request to SGLang endpoint
        url = f"{self.sglang_url.rstrip('/')}/generate"
        payload = {
            "input_embeds": v_scaled[0].cpu().tolist(),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("text", "")
        return {
            "explanation": self.extract_explanation_tags(raw_text),
            "raw_text": raw_text,
            "injection_status": "sglang_success",
        }

    def _verbalize_heuristic(self, v_scaled: torch.Tensor) -> Dict[str, Any]:
        """Standalone descriptor used for unit testing, mocks, or local development."""
        norm_val = float(v_scaled.norm().item())
        top_indices = torch.topk(v_scaled.abs(), k=min(3, v_scaled.shape[-1])).indices[0].tolist()
        desc = (
            f"Active feature direction concentrated along latent dimensions {top_indices} "
            f"with calibrated norm {norm_val:.1f}."
        )
        return {
            "explanation": desc,
            "raw_text": f"<explanation>{desc}</explanation>",
            "injection_status": "standalone_heuristic",
        }
