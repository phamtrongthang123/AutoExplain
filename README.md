# AutoExplain

AutoExplain is a minimal, persistent XAI browser harness for neural networks.

Instead of writing one-off forward hooks for every experiment, you control model execution through a browser-style automation interface. A single persistent session loads model weights once and supports repeated inspect-observe-intervene cycles without reinitialization.

## Public API Actions

| Action | Responsibility |
|---|---|
| `open_model()` | Load a Transformer into memory |
| `inspect()` | Inspect architecture, layers, and heads |
| `run()` | Run model inference |
| `observe()` | Inspect internal activations, attention, and logits |
| `intervene()` | Modify internal activations |
| `reset()` | Restore baseline behavior |
| `evaluate()` | Compare baseline and modified behavior |
| `record()` | Save experiment results |

## Quick Start

```python
from autoexplain import Harness

h = Harness()
h.open_model("gpt2")

h.inspect()

baseline = h.run("Alice gave Bob a book.")
h.observe(layer=5)

h.intervene(layer=5, head=3, method="zero")
modified = h.run("Alice gave Bob a book.")

h.reset()

result = h.evaluate(baseline, modified)
h.record("experiment_001")
```

Run the included GPT-2 demonstration:

```bash
python examples/gpt2.py
```

## How It Works

- **Persistent Model Instance**: `open_model("gpt2")` loads the Transformer into memory once. Model weights persist across runs, interventions, and evaluations.
- **Genuine Head Interventions**: `intervene(layer=5, head=3, method="zero")` configures forward hooks on the attention head output tensor (`hook_z`).
- **Numerical Evaluation**: `evaluate(baseline, modified)` measures actual output differences, including max logit difference, mean difference, cosine similarity, and KL divergence.
- **Fast Reset**: `reset()` clears active intervention hooks without reloading weights, restoring baseline outputs within float precision.
- **Experiment Records**: `record(id)` writes the prompt, intervention parameters, evaluation metrics, and top token predictions to a JSON file.

## Installation

```bash
git clone https://github.com/phamtrongthang123/AutoExplain.git
cd AutoExplain
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

## License

MIT
