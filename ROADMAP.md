# AutoExplain — Feature Roadmap v0.1

AutoExplain is designed as a browser-style automation harness for neural networks. Each interpretability technique contributes an action or method, unified behind a persistent execution harness.

## 1. Core Architecture

AutoExplain provides eight actions on a loaded model:

```
                  ┌──────────────────────────────────────────────┐
                  │              AutoExplain Harness             │
                  │ open_model · inspect · run · observe         │
                  │ intervene  · reset   · evaluate · record     │
                  └──────────────────────┬───────────────────────┘
                                         │
        ┌───────────────┬────────────────┼───────────────┬───────────────┐
        │               │                │               │               │
        ▼               ▼                ▼               ▼               ▼
   ┌─────────┐    ┌───────────┐    ┌───────────┐   ┌───────────┐   ┌───────────┐
   │   SAE   │    │  J-Space  │    │    NLA    │   │Block-Spar.│   │ Manifold  │ ... (Methods)
   │ Sparse  │    │ Jacobian  │    │Activation │   │Multidim.  │   │ Steering  │
   │Features │    │  Readouts │    │  Language │   │ Concepts  │   │ Intervene │
   └─────────┘    └───────────┘    └───────────┘   └───────────┘   └───────────┘
```

## 2. Feature Milestones

| Stage | Milestone | Status | Deliverable |
| :--- | :--- | :---: | :--- |
| **v0.1** | Persistent XAI Harness | Complete | Single-instance session (`open_model`, `inspect`, `run`, `observe`, `intervene`, `reset`, `evaluate`, `record`) |
| **F1** | Sparse Autoencoders (SAE) | Planned | Discover and ablate sparse dictionary features |
| **F2** | J-Space | Planned | Read out internal representations using the Jacobian Lens |
| **F3** | Agentic Investigation | Planned | Agent selects harness actions and tests hypotheses within budget |
| **F4** | Natural Language Autoencoders (NLA) | Standalone Module | Translate activations to natural language and score fidelity (kitft checkpoints preserved) |
| **F5** | Block-Sparse Featurizers | Planned | Discover multidimensional visual concepts |
| **F6** | Manifold Steering | Planned | Intervene along learned representation geometry |
| **F7** | AutoRefine | Planned | Propose, apply, and verify controlled model modifications |

## 3. v0.1 Harness Implementation Checklist

- [x] Load Transformer into memory once (`open_model`)
- [x] Inspect architecture, layers, and heads (`inspect`)
- [x] Run model inference with prompt caching (`run`)
- [x] Retrieve internal activation summaries without giant tensor transfers (`observe`)
- [x] Genuine attention-head ablation via forward hooks (`intervene`)
- [x] Restore baseline outputs without reloading model weights (`reset`)
- [x] Compute numerical differences between baseline and modified outputs (`evaluate`)
- [x] Save experiment configuration and measured results as JSON (`record`)
- [x] Working GPT-2 demonstration (`examples/gpt2.py`)
- [x] Acceptance tests proving all criteria (`tests/test_harness.py`)
