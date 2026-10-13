"""Investigation report container and HTML/Markdown generators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from autoexplain.core.experiment import ExperimentResult


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoExplain Investigation Report</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 2rem;
    }
    .container { max-width: 1000px; margin: 0 auto; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
    h1 { color: var(--accent); margin: 0 0 0.5rem 0; font-size: 2rem; }
    .meta { color: var(--muted); font-size: 0.9rem; }
    .badge {
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-success { background: rgba(52, 211, 153, 0.2); color: var(--success); }
    .badge-danger { background: rgba(248, 113, 113, 0.2); color: var(--danger); }
    .badge-warning { background: rgba(251, 191, 36, 0.2); color: var(--warning); }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .card h2 { margin-top: 0; color: var(--accent); font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
    ul { padding-left: 1.2rem; }
    li { margin-bottom: 0.5rem; }
    pre {
      background: #090d16;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
      border: 1px solid var(--border);
    }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    th, td { text-align: left; padding: 0.75rem; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>AutoExplain Investigation Report</h1>
      <div class="meta">
        <strong>Model:</strong> {{ model_name }} &nbsp;|&nbsp;
        <strong>Generated:</strong> {{ timestamp }} &nbsp;|&nbsp;
        <strong>Question:</strong> {{ question }}
      </div>
    </header>

    <div class="card">
      <h2>Summary</h2>
      <p>{{ summary or "No summary provided." }}</p>
    </div>

    <div class="card">
      <h2>Hypotheses & Status</h2>
      <table>
        <thead>
          <tr>
            <th>Hypothesis</th>
            <th>Status</th>
            <th>Causal Verification</th>
          </tr>
        </thead>
        <tbody>
          {% for h in hypotheses %}
          <tr>
            <td>{{ h.statement }}</td>
            <td>
              <span class="badge {% if h.status == 'supported' %}badge-success{% elif h.status == 'refuted' %}badge-danger{% else %}badge-warning{% endif %}">
                {{ h.status }}
              </span>
            </td>
            <td>{{ h.causal_test_summary }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Supporting Evidence</h2>
      <ul>
        {% for ev in evidence %}
        <li>{{ ev }}</li>
        {% else %}
        <li>No verified evidence recorded.</li>
        {% endfor %}
      </ul>
    </div>

    <div class="card">
      <h2>Contradictions and Limitations</h2>
      <ul>
        {% for con in contradictions %}
        <li>{{ con }}</li>
        {% else %}
        <li>No contradictions identified.</li>
        {% endfor %}
      </ul>
    </div>

    <div class="card">
      <h2>Recommended Next Experiments</h2>
      <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% else %}
        <li>None.</li>
        {% endfor %}
      </ul>
    </div>
  </div>
</body>
</html>
"""


@dataclass
class Hypothesis:
    """A scientific hypothesis generated during investigation."""

    statement: str
    status: str = "pending"  # "supported", "refuted", "inconclusive", "pending"
    causal_test_summary: str = "Awaiting verification"


@dataclass
class InvestigationReport:
    """Comprehensive investigation report synthesizing findings across tools."""

    question: str
    model_name: str
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hypotheses: List[Hypothesis] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    experiments: List[ExperimentResult] = field(default_factory=list)

    def save_html(self, filepath: str | Path) -> Path:
        """Render and save self-contained HTML report."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        template = Template(HTML_REPORT_TEMPLATE)
        html = template.render(
            question=self.question,
            model_name=self.model_name,
            summary=self.summary,
            timestamp=self.timestamp,
            hypotheses=[asdict(h) for h in self.hypotheses],
            evidence=self.evidence,
            contradictions=self.contradictions,
            recommendations=self.recommendations,
            experiments=[e.to_dict() for e in self.experiments],
        )
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def to_markdown(self) -> str:
        """Export report to formatted GitHub Markdown."""
        lines = [
            f"# AutoExplain Report: {self.question}",
            f"- **Model:** {self.model_name}",
            f"- **Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            self.summary or "N/A",
            "",
            "## Hypotheses",
        ]
        for h in self.hypotheses:
            lines.append(f"- **[{h.status.upper()}]** {h.statement} (Causal check: {h.causal_test_summary})")

        lines.extend(["", "## Evidence"])
        for ev in self.evidence:
            lines.append(f"- {ev}")

        lines.extend(["", "## Contradictions and Limitations"])
        for con in self.contradictions:
            lines.append(f"- {con}")

        lines.extend(["", "## Recommended Next Steps"])
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)
