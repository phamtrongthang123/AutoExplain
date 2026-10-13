"""PyTorch Model Adapter for activations, gradients, and causal interventions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ModelAdapter:
    """Universal PyTorch model adapter for interpretability research.

    Provides safe activation caching, gradient extraction, and causal interventions.
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float32

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM
            self.model_name = model
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
        else:
            self.model_name = model.__class__.__name__
            self.model = model.to(self.device)

        self.model.eval()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._gradient_cache: Dict[str, torch.Tensor] = {}

    def get_submodule(self, target_name: str) -> nn.Module:
        """Resolve a dotted layer path (e.g. 'transformer.h.0.mlp')."""
        return self.model.get_submodule(target_name)

    def list_layer_names(self) -> List[str]:
        """Return all named module paths in the model."""
        return [name for name, _ in self.model.named_modules() if name]

    @contextmanager
    def capture_activations(
        self,
        layer_names: List[str],
        detach: bool = True,
        clone: bool = True,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Context manager to record layer forward activations."""
        cache: Dict[str, torch.Tensor] = {}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def make_hook(name: str):
            def hook_fn(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
                tensor = output[0] if isinstance(output, tuple) else output
                if detach:
                    tensor = tensor.detach()
                if clone:
                    tensor = tensor.clone()
                cache[name] = tensor

            return hook_fn

        try:
            for name in layer_names:
                submod = self.get_submodule(name)
                handles.append(submod.register_forward_hook(make_hook(name)))
            yield cache
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def intervene_activation(
        self,
        layer_name: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Iterator[None]:
        """Context manager to causally perturb layer activations during forward pass.

        Args:
            layer_name: Dotted submodule name to intercept.
            intervention_fn: Callable that accepts the layer output tensor and returns the modified tensor.
        """
        submod = self.get_submodule(layer_name)

        def intervention_hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
            if isinstance(output, tuple):
                mod_out = intervention_fn(output[0])
                return (mod_out, *output[1:])
            return intervention_fn(output)

        handle = submod.register_forward_hook(intervention_hook)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)
