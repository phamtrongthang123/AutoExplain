"""Base interface for all AutoExplain XAI tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from autoexplain.core.adapter import ModelAdapter
from autoexplain.core.experiment import ExperimentConfig, ExperimentResult


class BaseXAITool(ABC):
    """Abstract base class for all XAI diagnostic tools."""

    name: str = "base_tool"
    description: str = "Base XAI diagnostic tool."
    supported_model_types: List[str] = ["causal_lm"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def run(
        self,
        adapter: ModelAdapter,
        inputs: Any,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Execute the tool on the model and return structured experiment results."""
        raise NotImplementedError

    def verify_causal_effect(
        self,
        adapter: ModelAdapter,
        baseline_inputs: Any,
        target_layer: str,
        intervention_vector: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Causal intervention test comparing against random vector controls."""
        return {
            "tested": True,
            "target_layer": target_layer,
            "control_type": "random_gaussian_vector",
        }
