"""Example: Autonomous Investigation of Model Latents using Natural Language Autoencoders (NLA)."""

import torch
import torch.nn as nn

from autoexplain import Investigator
from autoexplain.tools.registry import ToolRegistry


class ModelWithLatentConcepts(nn.Module):
    """Demonstration PyTorch model with structured latent representations."""

    def __init__(self, vocab_size: int = 100, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layer_0 = nn.Linear(d_model, d_model)
        self.layer_1 = nn.Linear(d_model, d_model)
        self.layer_2 = nn.Linear(d_model, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = h + torch.relu(self.layer_0(h))
        h = h + torch.relu(self.layer_1(h))
        h = h + torch.relu(self.layer_2(h))
        return self.output_head(h)


def run_demo():
    print("=== AutoExplain: Autonomous NLA Investigation Demo ===")

    # 1. Instantiate model and investigator agent
    model = ModelWithLatentConcepts()
    agent = Investigator(model=model, device="cpu")

    # 2. Formulate scientific inquiry
    question = "What internal concepts are encoded at layer_1 during inference?"
    print(f"Scientific Question: {question}")
    print(f"Available tools in registry: {ToolRegistry.list_tools()}")

    # 3. Launch investigation
    dummy_input = torch.tensor([[12, 45, 88, 3, 19]])
    report = agent.investigate(
        question=question,
        tools=["nla"],
        max_experiments=3,
        inputs=dummy_input,
    )

    # 4. Display findings
    print("\n--- Investigation Findings ---")
    for finding in report.evidence:
        print(f"  [Evidence] {finding}")

    if report.contradictions:
        print("\n--- Contradictions / Limitations Identified ---")
        for con in report.contradictions:
            print(f"  [Contradiction] {con}")

    # 5. Export interactive HTML report
    out_html = "nla_investigation_report.html"
    saved_path = report.save_html(out_html)
    print(f"\nInteractive HTML investigation report saved to: {saved_path}")


if __name__ == "__main__":
    run_demo()
