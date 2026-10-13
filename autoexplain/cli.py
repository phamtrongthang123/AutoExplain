"""Command-line interface for AutoExplain."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from autoexplain import __version__
from autoexplain.agent.investigator import Investigator
from autoexplain.tools.registry import ToolRegistry

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """AutoExplain: Autonomous Agentic Explainable AI Platform."""
    pass


@main.command(name="list-tools")
def list_tools():
    """List all registered XAI diagnostic tools."""
    table = Table(title="Available AutoExplain Diagnostic Tools")
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Supported Models", style="green")

    meta = ToolRegistry.get_tool_metadata()
    if not meta:
        console.print("[yellow]No tools currently registered in registry.[/yellow]")
        return

    for tool_id, info in meta.items():
        table.add_row(
            tool_id,
            info["description"],
            ", ".join(info["supported_model_types"]),
        )
    console.print(table)


@main.command(name="investigate")
@click.option("--model", required=True, help="HuggingFace model ID or local path.")
@click.option("--question", required=True, help="Investigation inquiry / prompt.")
@click.option("--tools", default="", help="Comma-separated list of tools to use.")
@click.option("--max-experiments", default=5, type=int, help="Maximum number of experiments.")
@click.option("--output", default="investigation_report.html", help="Path to save HTML report.")
def investigate(model: str, question: str, tools: str, max_experiments: int, output: str):
    """Run an autonomous investigation."""
    console.print(f"[bold cyan]AutoExplain Investigation Agent starting...[/bold cyan]")
    console.print(f"[bold]Target Model:[/bold] {model}")
    console.print(f"[bold]Inquiry:[/bold] {question}")

    tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else None

    agent = Investigator(model=model)
    report = agent.investigate(
        question=question,
        tools=tool_list,
        max_experiments=max_experiments,
    )

    saved_path = report.save_html(output)
    console.print(f"[bold green]Investigation complete.[/bold green] Report saved to [underline]{saved_path}[/underline]")


if __name__ == "__main__":
    main()
