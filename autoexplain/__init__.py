"""AutoExplain: Minimal, persistent XAI browser harness for neural networks."""

from autoexplain.experiment import ExperimentRecord, RunOutput
from autoexplain.harness import Harness

__version__ = "0.1.0"

__all__ = [
    "Harness",
    "RunOutput",
    "ExperimentRecord",
]
