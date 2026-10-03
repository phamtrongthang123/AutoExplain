"""Explicit two-stage demo: propose a plan, inspect it, then run locally."""
import argparse
import json
from pathlib import Path

from .research import CodexPlanner, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-demo", help="Calls Codex; sends only the question and built-in public summary")
    plan.add_argument("--question", required=True)
    plan.add_argument("--model", default="gpt-6-luna")
    plan.add_argument("--reasoning", default="high")
    plan.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run-demo", help="No LLM call; executes the reviewed JSON on a tiny local CNN")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan-demo":
        if args.out.exists():
            parser.error("Plan output already exists")
        proposal = CodexPlanner(model=args.model, reasoning=args.reasoning).propose(args.question, {
            "architecture": "BarNet two-layer CNN", "task": "vertical vs horizontal synthetic bars",
            "input_shape": [2, 1, 16, 16], "output_shape": [2, 2],
            "spatial_layers": ["features.0", "features.2"], "classes": ["vertical", "horizontal"],
            "device": "cpu", "weights": "trained locally, never supplied to planner"})
        with args.out.open("x") as file:
            json.dump(proposal, file, indent=2)
        print(f"Proposal saved to {args.out}. Review it before run-demo. No experiment executed.")
    else:
        import torch
        from .demo import trained_demo, bars
        torch.set_num_threads(2)
        proposal = json.loads(args.plan.read_text())
        model = trained_demo()
        inputs, _ = bars(2, seed=99)
        report = run_experiment(model, inputs, proposal, args.out)
        print(f"{report['status']}: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
