"""Experiment configuration and result data models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Standardized configuration for an XAI experiment."""

    tool_name: str
    target_layers: List[str] = field(default_factory=list)
    model_name: str = ""
    seed: int = 42
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ExperimentResult:
    """Standardized output container for an XAI experiment."""

    experiment_id: str
    tool_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: Optional[ExperimentConfig] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    causal_verification_passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment result to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize experiment result to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
