"""AutoExplain Harness: Minimal, persistent XAI browser harness for neural networks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from autoexplain.experiment import ExperimentRecord, RunOutput


class Harness:
    """Persistent execution harness for mechanistic interpretability on Transformers."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[HookedTransformer] = None
        self.model_name: Optional[str] = None

        # Persistent session state
        self._active_interventions: List[Dict[str, Any]] = []
        self._last_interventions: List[Dict[str, Any]] = []
        self._last_prompt: Optional[str] = None
        self._last_output: Optional[RunOutput] = None
        self._last_cache: Any = None
        self._last_baseline: Optional[RunOutput] = None
        self._last_modified: Optional[RunOutput] = None
        self._last_evaluation: Optional[Dict[str, Any]] = None

    def open_model(self, model_name: str = "gpt2", device: Optional[str] = None) -> Dict[str, Any]:
        """Load a Transformer model into memory, persisting weights across experiments."""
        target_device = device or self.device

        # Reuse existing loaded model if identical
        if self.model is not None and self.model_name == model_name:
            return self.inspect()

        self.model = HookedTransformer.from_pretrained(model_name, device=target_device)
        self.model_name = model_name
        self.device = target_device
        self.reset()
        return self.inspect()

    def inspect(self) -> Dict[str, Any]:
        """Inspect architecture, layers, and heads."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call open_model() first.")

        cfg = self.model.cfg
        n_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_name": self.model_name,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "d_mlp": cfg.d_mlp,
            "d_vocab": cfg.d_vocab,
            "n_params": n_params,
            "device": str(self.device),
            "layers": list(range(cfg.n_layers)),
            "heads_per_layer": cfg.n_heads,
        }

    def _build_fwd_hooks(self) -> List[Tuple[str, Callable]]:
        """Construct HookedTransformer hooks for active interventions."""
        hooks: List[Tuple[str, Callable]] = []

        for intervention in self._active_interventions:
            layer = intervention["layer"]
            head = intervention.get("head")
            method = intervention.get("method", "zero")
            value = intervention.get("value")

            if head is not None:
                # Intervene on specific attention head output (hook_z)
                hook_name = f"blocks.{layer}.attn.hook_z"

                def make_head_hook(target_head: int, m: str, v: Any):
                    def head_hook_fn(z: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                        out = z.clone()
                        if m == "zero":
                            out[:, :, target_head, :] = 0.0
                        elif m == "scale":
                            scale = float(v) if v is not None else 0.0
                            out[:, :, target_head, :] *= scale
                        elif m == "set":
                            out[:, :, target_head, :] = v.to(out.device, out.dtype)
                        return out

                    return head_hook_fn

                hooks.append((hook_name, make_head_hook(head, method, value)))
            else:
                # Intervene on layer residual output
                hook_name = f"blocks.{layer}.hook_resid_post"

                def make_layer_hook(m: str, v: Any):
                    def layer_hook_fn(resid: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                        out = resid.clone()
                        if m == "zero":
                            out[:] = 0.0
                        elif m == "scale":
                            scale = float(v) if v is not None else 0.0
                            out[:] *= scale
                        return out

                    return layer_hook_fn

                hooks.append((hook_name, make_layer_hook(method, value)))

        return hooks

    def run(self, prompt: str, max_new_tokens: int = 0) -> RunOutput:
        """Run model inference, applying any configured interventions."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call open_model() first.")

        hooks = self._build_fwd_hooks()

        if hooks:
            with self.model.hooks(fwd_hooks=hooks):
                logits, cache = self.model.run_with_cache(prompt)
        else:
            logits, cache = self.model.run_with_cache(prompt)

        self._last_prompt = prompt
        self._last_cache = cache
        if self._active_interventions:
            self._last_interventions = list(self._active_interventions)

        # Compute top predicted tokens at final position
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, k=5)

        top_tokens = []
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            tok_str = self.model.to_string([idx])
            top_tokens.append({
                "token_id": idx,
                "token": tok_str,
                "probability": float(p),
                "logit": float(last_logits[idx].item()),
            })

        output_text = prompt
        if max_new_tokens > 0:
            if hooks:
                with self.model.hooks(fwd_hooks=hooks):
                    output_text = self.model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)
            else:
                output_text = self.model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)

        run_out = RunOutput(
            prompt=prompt,
            text=output_text,
            logits=last_logits.detach().cpu(),
            top_tokens=top_tokens,
        )
        self._last_output = run_out
        return run_out

    def observe(self, layer: int, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Inspect internal activations, attention, and logits for the current prompt."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call open_model() first.")

        active_prompt = prompt or self._last_prompt
        if active_prompt is None:
            raise ValueError("No prompt provided and no previous run recorded.")

        # Re-run if cache is missing or prompt changed
        if self._last_cache is None or active_prompt != self._last_prompt:
            self.run(active_prompt)

        cache = self._last_cache
        tokens = self.model.to_str_tokens(active_prompt)

        # 1. Residual stream norm at layer output
        resid_key = f"blocks.{layer}.hook_resid_post"
        resid_norm = float(cache[resid_key].norm(dim=-1).mean().item()) if resid_key in cache else 0.0

        # 2. Per-head output norms
        head_key = f"blocks.{layer}.attn.hook_z"
        head_norms: Dict[int, float] = {}
        if head_key in cache:
            # hook_z is [batch, seq_len, n_heads, d_head]
            z = cache[head_key][0]  # [seq_len, n_heads, d_head]
            per_head = z.norm(dim=-1).mean(dim=0)  # [n_heads]
            for h_idx, norm_val in enumerate(per_head.tolist()):
                head_norms[h_idx] = float(norm_val)

        # 3. Attention block output norm
        attn_out_key = f"blocks.{layer}.hook_attn_out"
        attn_out_norm = float(cache[attn_out_key].norm(dim=-1).mean().item()) if attn_out_key in cache else 0.0

        # 4. MLP block output norm
        mlp_out_key = f"blocks.{layer}.hook_mlp_out"
        mlp_out_norm = float(cache[mlp_out_key].norm(dim=-1).mean().item()) if mlp_out_key in cache else 0.0

        top_predictions = self._last_output.top_tokens if self._last_output else []

        return {
            "layer": layer,
            "prompt": active_prompt,
            "tokens": tokens,
            "residual_norm": resid_norm,
            "attention_output_norm": attn_out_norm,
            "mlp_output_norm": mlp_out_norm,
            "head_norms": head_norms,
            "top_predictions": top_predictions,
        }

    def intervene(
        self,
        layer: int,
        head: Optional[int] = None,
        method: str = "zero",
        value: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Modify internal activations for subsequent runs."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call open_model() first.")

        cfg = {
            "layer": layer,
            "head": head,
            "method": method,
            "value": value,
        }
        self._active_interventions.append(cfg)
        return list(self._active_interventions)

    def reset(self) -> Harness:
        """Restore baseline behavior without reloading model weights."""
        self._active_interventions.clear()
        return self

    def evaluate(self, baseline: RunOutput, modified: RunOutput) -> Dict[str, Any]:
        """Compute numerical differences between baseline and modified outputs."""
        b_logits = baseline.logits.float()
        m_logits = modified.logits.float()

        # Logit differences
        logit_diff = (b_logits - m_logits).abs()
        max_diff = float(logit_diff.max().item())
        mean_diff = float(logit_diff.mean().item())

        # Cosine similarity
        cos_sim = float(F.cosine_similarity(b_logits.unsqueeze(0), m_logits.unsqueeze(0)).item())

        # KL Divergence: KL(P_baseline || P_modified)
        p = F.softmax(b_logits, dim=-1)
        q = F.softmax(m_logits, dim=-1)
        kl_div = float(F.kl_div(q.log(), p, reduction="sum").item())

        # Prediction shift
        b_top = baseline.top_tokens[0]["token"] if baseline.top_tokens else ""
        m_top = modified.top_tokens[0]["token"] if modified.top_tokens else ""
        pred_changed = (b_top != m_top)

        result = {
            "max_logit_diff": max_diff,
            "mean_logit_diff": mean_diff,
            "cosine_similarity": cos_sim,
            "kl_divergence": kl_div,
            "top_prediction_changed": pred_changed,
            "baseline_top_token": b_top,
            "modified_top_token": m_top,
            "baseline_probability": baseline.top_tokens[0]["probability"] if baseline.top_tokens else 0.0,
            "modified_probability": modified.top_tokens[0]["probability"] if modified.top_tokens else 0.0,
        }

        self._last_baseline = baseline
        self._last_modified = modified
        self._last_evaluation = result
        return result

    def record(self, experiment_id: str, path: Optional[Union[str, Path]] = None) -> Path:
        """Save experiment configuration and measured results as JSON."""
        target_path = Path(path) if path else Path("experiments") / f"{experiment_id}.json"

        interventions = list(self._active_interventions) if self._active_interventions else list(self._last_interventions)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            model=self.model_name or "unknown",
            prompt=self._last_prompt or "",
            interventions=interventions,
            evaluation=self._last_evaluation or {},
            baseline_top_tokens=self._last_baseline.top_tokens if self._last_baseline else [],
            modified_top_tokens=self._last_modified.top_tokens if self._last_modified else [],
        )

        saved = record.save(target_path)
        return saved
