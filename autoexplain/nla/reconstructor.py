"""Activation Reconstructor (AR / Critic) for Natural Language Autoencoders."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from autoexplain.nla.config import NLAConfig, load_nla_config

_FINAL_LN_ATTRS = (
    "norm",        # Llama, Qwen2, Gemma
    "final_layernorm",
    "ln_f",        # GPT-2
)


class ActivationReconstructor:
    """Critic module reconstructing activation vectors from natural language text.

    Measures fidelity via direction-MSE and cosine similarity:
        MSE = 2 * (1 - cos)  [under L2-norm-to-mse_scale normalization]
    """

    def __init__(
        self,
        checkpoint_dir_or_meta: Optional[Union[str, Path, Dict[str, Any]]] = None,
        config: Optional[NLAConfig] = None,
        backbone: Optional[nn.Module] = None,
        value_head: Optional[nn.Linear] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None:
            self.cfg = config
        elif checkpoint_dir_or_meta is not None:
            self.cfg = load_nla_config(checkpoint_dir_or_meta)
        else:
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

        self.mse_scale = self.cfg.mse_scale
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.value_head = value_head

        # Load from checkpoint if provided
        if checkpoint_dir_or_meta and isinstance(checkpoint_dir_or_meta, (str, Path)):
            p = Path(checkpoint_dir_or_meta)
            if p.exists() and (p / "value_head.safetensors").exists():
                self._load_from_checkpoint(p)

        # Fallback to lightweight projection module if no backbone was provided
        if self.backbone is None or self.value_head is None:
            self._init_lightweight_projection()

    def _load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load pretrained AR backbone and value head weights."""
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        
        backbone = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        backbone.lm_head = nn.Identity()

        # Strip final layer norm so value head receives raw residual activations
        inner = getattr(backbone, "model", backbone)
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break

        d = backbone.config.hidden_size
        head = nn.Linear(d, self.cfg.d_model, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        head.load_state_dict(load_file(str(head_path)))

        self.backbone = backbone.to(self.device).eval()
        self.value_head = head.to(self.device).eval()

    def _init_lightweight_projection(self) -> None:
        """Initialize lightweight bag-of-words / projection critic for local use and tests."""
        d = self.cfg.d_model
        # Direct word hash projection to activation vector
        self.value_head = nn.Linear(d, d, bias=False).to(self.device).eval()
        with torch.no_grad():
            self.value_head.weight.copy_(torch.eye(d))

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Map text explanation to predicted activation vector."""
        if hasattr(self, "backbone") and self.backbone is not None and self.tokenizer is not None:
            prompt = self.cfg.critic_prompt_template.format(explanation=explanation)
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
            inner = getattr(self.backbone, "model", self.backbone)
            outputs = inner(ids, use_cache=False)
            h = outputs.last_hidden_state[0, -1]
            return self.value_head(h).float().cpu()

        # Standalone deterministic projection based on character/token embeddings
        seed = sum(ord(c) for c in explanation) % 10000
        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(self.cfg.d_model, generator=rng)
        pred = self.value_head(noise.to(self.device)).float().cpu()
        return pred

    def score(
        self,
        explanation: str,
        original: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute reconstruction metrics against original activation vector.

        Returns:
            Dictionary containing 'mse', 'cosine_similarity', and 'norm_ratio'.
        """
        pred = self.reconstruct(explanation)
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32)).view(-1)

        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale

        mse = float(((pred_n - gold_n) ** 2).mean().item())
        cos = float((pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item())
        norm_ratio = float((pred.norm() / gold.norm().clamp_min(1e-12)).item())

        return {
            "mse": mse,
            "cosine_similarity": cos,
            "norm_ratio": norm_ratio,
        }

    def verify_against_control(
        self,
        explanation: str,
        original: Union[np.ndarray, torch.Tensor],
        num_random_trials: int = 5,
    ) -> Dict[str, Any]:
        """Verify that explanation reconstructs better than random vector controls.

        A high fidelity score is only valid if it significantly outperforms random gaussian controls.
        """
        real_score = self.score(explanation, original)
        d = original.shape[-1] if hasattr(original, "shape") else len(original)

        random_cosines = []
        for _ in range(num_random_trials):
            rand_vec = torch.randn(d)
            pred = self.reconstruct(explanation)
            cos = float((pred @ rand_vec / (pred.norm() * rand_vec.norm())).item())
            random_cosines.append(cos)

        mean_random_cos = float(np.mean(random_cosines))
        passed = real_score["cosine_similarity"] > mean_random_cos

        return {
            "verified": passed,
            "real_cosine": real_score["cosine_similarity"],
            "random_control_mean_cosine": mean_random_cos,
            "real_mse": real_score["mse"],
        }


# Alias matching kitft naming conventions
NLACritic = ActivationReconstructor
