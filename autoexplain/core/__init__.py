"""Core infrastructure for AutoExplain."""

from autoexplain.core.adapter import ModelAdapter
from autoexplain.core.experiment import ExperimentConfig, ExperimentResult

__all__ = ["ModelAdapter", "ExperimentConfig", "ExperimentResult"]
