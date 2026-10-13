"""Unit tests for AutoExplain core infrastructure."""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn

from autoexplain.core.adapter import ModelAdapter
from autoexplain.core.experiment import ExperimentConfig, ExperimentResult
from autoexplain.reporting.report import Hypothesis, InvestigationReport
from autoexplain.tools.base import BaseXAITool
from autoexplain.tools.registry import ToolRegistry, register_tool


class TinyMLP(nn.Module):
    """Tiny 2-layer MLP for unit tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_model_adapter_activation_capture():
    model = TinyMLP()
    adapter = ModelAdapter(model=model, device="cpu")

    x = torch.randn(2, 8)
    with adapter.capture_activations(["fc1", "fc2"]) as cache:
        out = adapter.forward(x)

    assert "fc1" in cache
    assert "fc2" in cache
    assert cache["fc1"].shape == (2, 16)
    assert cache["fc2"].shape == (2, 2)
    assert out.shape == (2, 2)


def test_model_adapter_intervention():
    model = TinyMLP()
    adapter = ModelAdapter(model=model, device="cpu")

    x = torch.randn(2, 8)
    baseline_out = adapter.forward(x).clone()

    def zero_out(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor)

    with adapter.intervene_activation("fc1", zero_out):
        intervened_out = adapter.forward(x)

    # Intervening fc1 should change output
    assert not torch.allclose(baseline_out, intervened_out)


def test_experiment_result_serialization():
    cfg = ExperimentConfig(tool_name="test_tool", target_layers=["fc1"])
    result = ExperimentResult(
        experiment_id="exp-001",
        tool_name="test_tool",
        config=cfg,
        findings=["Found high activation feature #4"],
    )
    d = result.to_dict()
    assert d["experiment_id"] == "exp-001"
    json_str = result.to_json()
    assert "Found high activation feature #4" in json_str


def test_report_generation_html_and_markdown():
    report = InvestigationReport(
        question="What controls output logits?",
        model_name="TinyMLP",
        summary="Test investigation completed successfully.",
        hypotheses=[
            Hypothesis(statement="fc1 node 2 controls class 0", status="supported", causal_test_summary="Passed ablation")
        ],
        evidence=["fc1 node 2 has 3.4x higher activation on target inputs"],
        contradictions=["Random noise control failed to trigger class 0"],
        recommendations=["Investigate fc2 weights connected to node 2"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "report.html"
        saved = report.save_html(out_html)
        assert saved.exists()
        content = saved.read_text(encoding="utf-8")
        assert "AutoExplain Investigation Report" in content
        assert "fc1 node 2 controls class 0" in content

    md = report.to_markdown()
    assert "# AutoExplain Report" in md
    assert "[SUPPORTED]" in md
