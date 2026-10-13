"""NLATool: Diagnostic XAI tool wrapping Natural Language Autoencoders."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch

from autoexplain.core.adapter import ModelAdapter
from autoexplain.core.experiment import ExperimentConfig, ExperimentResult
from autoexplain.nla.config import NLAConfig
from autoexplain.nla.reconstructor import ActivationReconstructor
from autoexplain.nla.verbalizer import ActivationVerbalizer
from autoexplain.tools.base import BaseXAITool
from autoexplain.tools.registry import register_tool

logger = logging.getLogger(__name__)


class NaturalLanguageAutoencoder:
    """Unified container joining an Activation Verbalizer (Actor) and Reconstructor (Critic)."""

    def __init__(
        self,
        verbalizer: Optional[ActivationVerbalizer] = None,
        reconstructor: Optional[ActivationReconstructor] = None,
        config: Optional[NLAConfig] = None,
    ) -> None:
        self.config = config
        self.verbalizer = verbalizer or ActivationVerbalizer(config=config)
        self.reconstructor = reconstructor or ActivationReconstructor(config=config)

    def verbalize(self, v: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        """Convert activation vector to natural language explanation."""
        return self.verbalizer.verbalize(v, **kwargs)

    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Convert natural language explanation back to activation vector."""
        return self.reconstructor.reconstruct(explanation)

    def autoencode(
        self,
        activation: torch.Tensor,
        num_control_trials: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute full round-trip: activation -> explanation -> reconstructed activation -> score."""
        verb_res = self.verbalize(activation, **kwargs)
        explanation = verb_res["explanation"]

        scores = self.reconstructor.score(explanation, activation)
        control_res = self.reconstructor.verify_against_control(
            explanation, activation, num_random_trials=num_control_trials
        )

        return {
            "explanation": explanation,
            "raw_text": verb_res.get("raw_text", ""),
            "injection_status": verb_res.get("injection_status", ""),
            "mse": scores["mse"],
            "cosine_similarity": scores["cosine_similarity"],
            "norm_ratio": scores["norm_ratio"],
            "control_verification": control_res,
        }


@register_tool("nla")
class NLATool(BaseXAITool):
    """Natural Language Autoencoder diagnostic tool for explaining internal representations."""

    name: str = "nla"
    description: str = (
        "Translates internal neural activations into natural-language descriptions "
        "and measures round-trip reconstruction fidelity (MSE / Cosine Similarity)."
    )
    supported_model_types: List[str] = ["causal_lm"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self.nla: Optional[NaturalLanguageAutoencoder] = None

    def _resolve_target_layer(self, adapter: ModelAdapter, layer_override: Optional[str] = None) -> str:
        """Select target layer, defaulting to approximately 2/3 deep into the model."""
        if layer_override:
            return layer_override

        all_layers = adapter.list_layer_names()
        # Filter for typical transformer block names
        block_candidates = [
            l for l in all_layers
            if any(term in l for term in ("layers.", "h.", "blocks."))
            and not any(sub in l for sub in ("self_attn", "mlp", "ln", "norm"))
        ]

        if block_candidates:
            target_idx = int(len(block_candidates) * 0.67)
            return block_candidates[min(target_idx, len(block_candidates) - 1)]

        # Fallback to the last available layer
        return all_layers[-1]

    def run(
        self,
        adapter: ModelAdapter,
        inputs: Any,
        target_layer: Optional[str] = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Capture activations, run NLA verbalization + reconstruction, and return experiment results."""
        resolved_layer = self._resolve_target_layer(adapter, target_layer)
        exp_cfg = ExperimentConfig(
            tool_name=self.name,
            target_layers=[resolved_layer],
            model_name=adapter.model_name,
            description=f"Natural Language Autoencoder decode on layer {resolved_layer}",
        )

        # Prepare sample forward inputs if not provided
        if inputs is None:
            # Generate dummy input matching adapter device
            dummy_ids = torch.tensor([[101, 2054, 2003, 1037, 102]], device=adapter.device)
            inputs = {"input_ids": dummy_ids}
        elif isinstance(inputs, str):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(adapter.model_name)
            inputs = tok(inputs, return_tensors="pt").to(adapter.device)

        # 1. Capture activation vector at target layer
        with adapter.capture_activations([resolved_layer]) as cache:
            if isinstance(inputs, dict):
                adapter.forward(**inputs)
            elif isinstance(inputs, torch.Tensor):
                adapter.forward(inputs)

        raw_act = cache[resolved_layer]
        # Pool to 1D vector (take last token position)
        if raw_act.ndim == 3:
            v_act = raw_act[0, -1]
        elif raw_act.ndim == 2:
            v_act = raw_act[0]
        else:
            v_act = raw_act.view(-1)

        d_model = v_act.shape[-1]

        # 2. Initialize NLA engine calibrated to this activation dimension
        if self.nla is None or self.nla.verbalizer.cfg.d_model != d_model:
            nla_cfg = NLAConfig(
                d_model=d_model,
                injection_char="㈎",
                injection_token_id=149705,
                injection_left_neighbor_id=29,
                injection_right_neighbor_id=522,
                actor_prompt_template="Explain activation: <concept>{injection_char}</concept>",
                critic_prompt_template="Summary of text: <text>{explanation}</text> <summary>",
                injection_scale=150.0,
                mse_scale=float(math.sqrt(d_model)),
            )
            self.nla = NaturalLanguageAutoencoder(config=nla_cfg)

        # 3. Perform round-trip autoencoding
        nla_res = self.nla.autoencode(v_act)
        cos_sim = nla_res["cosine_similarity"]
        mse = nla_res["mse"]
        control_passed = nla_res["control_verification"]["verified"]

        findings: List[str] = [
            f"NLA Verbalization for layer '{resolved_layer}': \"{nla_res['explanation']}\"",
            f"Reconstruction fidelity: Cosine Similarity = {cos_sim:.3f}, Direction-MSE = {mse:.3f}.",
        ]

        contradictions: List[str] = []
        if not control_passed:
            contradictions.append(
                f"Fidelity failed random vector control: real cosine ({cos_sim:.3f}) did not "
                f"exceed random noise baseline ({nla_res['control_verification']['random_control_mean_cosine']:.3f})."
            )
        if cos_sim < 0.3:
            contradictions.append(
                f"Low semantic fidelity (cos={cos_sim:.3f} < 0.30): verbalization may be speculative or under-specified."
            )

        return ExperimentResult(
            experiment_id=f"exp-nla-{resolved_layer.replace('.', '_')}",
            tool_name=self.name,
            config=exp_cfg,
            metrics={
                "cosine_similarity": cos_sim,
                "mse": mse,
                "norm_ratio": nla_res["norm_ratio"],
            },
            findings=findings,
            contradictions=contradictions,
            raw_outputs=nla_res,
            causal_verification_passed=(control_passed and cos_sim >= 0.5),
        )
