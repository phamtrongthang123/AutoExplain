"""AutoExplain: Autonomous Agentic Explainable AI (XAI) Platform."""

from autoexplain.agent.investigator import Investigator
from autoexplain.core.adapter import ModelAdapter
from autoexplain.nla import NLATool, NaturalLanguageAutoencoder
from autoexplain.reporting.report import InvestigationReport
from autoexplain.tools.registry import ToolRegistry, register_tool

__version__ = "0.1.0a0"

__all__ = [
    "Investigator",
    "ModelAdapter",
    "InvestigationReport",
    "ToolRegistry",
    "register_tool",
    "NaturalLanguageAutoencoder",
    "NLATool",
]
