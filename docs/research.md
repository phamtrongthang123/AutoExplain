# Codex-assisted local experiments

The optional planner uses your authenticated **Codex CLI**, defaulting to
`gpt-6-luna` with `high` reasoning. The CLI model cache on the development machine
lists that model and effort; availability on other accounts is not guaranteed.
No separate API key is managed by AutoExplain. Calls consume your Codex quota.

## Propose, inspect, execute

From an installed checkout:

```bash
python -m autoexplain.research_cli plan-demo \
  --question "Do gradient-based rankings outperform random pixel deletion for the vertical-bar score?" \
  --model gpt-6-luna --reasoning high --out proposed-plan.json
```

Inspect `proposed-plan.json` before execution. It contains a question, unverified
hypothesis, allowlisted methods, target class, spatial layer and limitations.
The planning command does not execute an experiment.

```bash
python -m autoexplain.research_cli run-demo \
  --plan proposed-plan.json --out experiment-001
```

The second command makes no LLM calls. It trains the tiny CNN locally, runs the
selected methods, and writes `plan.json`, `report.json` and `attributions.pt`.
Reports include deletion curves, matched random-order controls, runtime and IG
completeness residuals when applicable. A failed execution writes a failed report
rather than a successful finding. Existing run directories are never overwritten.

## Your own model

```python
from autoexplain.research import CodexPlanner, run_experiment

# This summary is explicitly transmitted. Do not include private samples.
proposal = CodexPlanner().propose(
    "Which input regions affect class 0?",
    {"task": "image classification", "input_shape": [1, 3, 16, 16],
     "output_shape": [1, 2], "spatial_layers": ["features.2"]},
)
print(proposal)  # Review separately before running.
# report = run_experiment(your_cpu_model, preprocessed_batch, proposal, "run-001")
```

The initial runner accepts CPU floating inputs, at most four samples and 4096
input elements. The model must return `[batch, classes]` scores. It does not infer
preprocessing, validate semantic suitability or sandbox your model code. Fixed
budgets are 16 IG steps, eight SmoothGrad samples and eight deletion intervals
with three random-order controls. Occlusion visits scalar input positions.
These limits bound method work but not the runtime or memory of an arbitrary
model's forward pass. This is a small classification workflow, not yet a generic
agent for all supported tutorial families.

## Privacy and execution boundary

AutoExplain builds prompts from the question and summary you explicitly supply;
it does not serialize or attach weights or datasets. There is no automatic
redaction. A later evidence-review integration would need explicit control over
which local results are shared; the present runner never sends reports back.

Codex executes in a temporary directory with read-only sandbox mode, ignored user
configuration/rules, ephemeral sessions, and instructions not to use tools.
**Those instructions are not a hard tool-disable mechanism, and a read-only
sandbox does not prevent file reads.** This wrapper is not an OS-level privacy
boundary. For sensitive or clinical work, run the planner in an appropriately
isolated environment or supply a plan manually without invoking Codex. Custom
user providers and MCP settings are intentionally not inherited via user config;
Codex authentication is still used. No approval bypass flags are enabled.

Plans are schema-checked and independently validated before local execution.
Only built-in methods can run; generated Python, shell commands and arbitrary
parameter dictionaries are not accepted. Failed calls are not automatically
retried. The planner has a 180-second timeout; service availability is external.

## What is and is not established

Mocked CLI tests verify command construction, model selection, JSON validation,
timeouts and failure handling. Local tests execute all six methods and check
artifacts. A live GPT-6-Luna request has not yet been validated by this integration.
The local model cache is not proof of successful remote inference.

Hypotheses remain marked `not_adjudicated` even after execution. A lower deletion
AUC is not automatically a scientific finding; baseline choice, distribution
shift, uncertainty and independent examples matter. Literature retrieval,
automatic hypothesis revision, cross-model adapters and persistent claim-ledger
review are still future work.
