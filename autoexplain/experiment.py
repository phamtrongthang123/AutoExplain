"""Experiment records and serialization for AutoExplain Harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Persistent record of an intervention experiment."""

    experiment_id: str
    model: str
    prompt: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    baseline_top_tokens: List[Dict[str, Any]] = field(default_factory=list)
    modified_top_tokens: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert record to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, target_path: str | Path) -> Path:
        """Write record to JSON file on disk."""
        out = Path(target_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out


@dataclass
class RunOutput:
    """Structured output from a model run."""

    prompt: str
    text: str
    logits: Any  # torch.Tensor containing final position logits
    top_tokens: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    loss: Optional[float] = None

    def __repr__(self) -> str:
        top_str = ", ".join(f"{t['token']}: {t['probability']:.3f}" for t in self.top_tokens[:3])
        return f"<RunOutput prompt={self.prompt!r} top=[{top_str}] interventions={len(self.interventions)}>"
