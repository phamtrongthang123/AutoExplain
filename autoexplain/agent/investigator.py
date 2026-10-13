"""Autonomous investigator agent orchestrating XAI tools."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from autoexplain.core.adapter import ModelAdapter
from autoexplain.reporting.report import Hypothesis, InvestigationReport
from autoexplain.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Investigator:
    """Agentic XAI Investigator orchestrating hypotheses, tools, and causal verification."""

    def __init__(
        self,
        model: Union[nn.Module, str],
        device: Optional[str] = None,
    ) -> None:
        self.adapter = ModelAdapter(model, device=device)
        self.registry = ToolRegistry

    def investigate(
        self,
        question: str,
        tools: Optional[List[str]] = None,
        max_experiments: int = 5,
        budget_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> InvestigationReport:
        """Run an autonomous hypothesis-driven investigation."""
        selected_tools = tools or self.registry.list_tools()
        report = InvestigationReport(
            question=question,
            model_name=self.adapter.model_name,
        )

        # Formulate initial hypotheses
        report.hypotheses.append(
            Hypothesis(
                statement=f"Specific sparse latent directions are responsible for: '{question}'",
                status="pending",
            )
        )

        report.summary = (
            f"Autonomous investigation initialized for model '{self.adapter.model_name}' "
            f"addressing: '{question}'. Tools selected: {selected_tools}."
        )

        # Execute experiments within budget
        experiments_run = 0
        for tool_name in selected_tools:
            if experiments_run >= max_experiments:
                break

            try:
                tool_cls = self.registry.get(tool_name)
                tool_instance = tool_cls()
                result = tool_instance.run(self.adapter, inputs=kwargs.get("inputs", None))
                report.experiments.append(result)
                report.evidence.extend(result.findings)
                report.contradictions.extend(result.contradictions)
                experiments_run += 1
            except Exception as e:
                report.contradictions.append(
                    f"Tool execution for '{tool_name}' failed with error: {str(e)}"
                )

        if not report.evidence:
            report.recommendations.append(
                "Run baseline activation capture across deeper MLP layers to locate concept candidates."
            )
        else:
            report.recommendations.append(
                "Perform causal steering along discovered top feature directions."
            )

        return report
