# AutoExplain — Feature Roadmap v0.1

**Vision:** Build a general-purpose agentic Explainable AI (XAI) product that can autonomously investigate, explain, test, and eventually refine deep-learning models.

AutoExplain is designed as a scientific suite of interpretability tools orchestrated by an AI agent. Each interpretability methodology contributes a capability to the ecosystem, unified behind a common experiment interface so any researcher or practitioner can investigate models without manual paper reimplementation.

---

## 1. Product Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │              AutoExplain Agent               │
                  │  Hypotheses · Tool Selection · Experiments   │
                  │        Pattern Discovery · Next Steps        │
                  └──────────────────────┬───────────────────────┘
                                         │
        ┌───────────────┬────────────────┼───────────────┬───────────────┐
        │               │                │               │               │
        ▼               ▼                ▼               ▼               ▼
   ┌─────────┐    ┌───────────┐    ┌───────────┐   ┌───────────┐   ┌───────────┐
   │   SAE   │    │  J-Space  │    │    NLA    │   │Block-Spar.│   │ Manifold  │ ... (Registry)
   │ Sparse  │    │ Jacobian  │    │Activation │   │Multidim.  │   │ Steering  │
   │Features │    │  Readouts │    │  Language │   │ Concepts  │   │ Intervene │
   └────┬────┘    └─────┬─────┘    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
        │               │                │               │               │
        └───────────────┴────────────────┼───────────────┴───────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             Investigation Report             │
                  │   Observations · Evidence · Contradictions   │
                  │         Causal Tests · Recommended Fixes     │
                  └──────────────────────┬───────────────────────┘
                                         │ (Future)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             AutoRefine (Future)              │
                  │      Model Modification & Verification       │
                  └──────────────────────────────────────────────┘
```

---

## 2. Feature Roadmap Milestones

| Stage | Feature | Product Deliverable |
| :--- | :--- | :--- |
| **F0** | Core XAI Infrastructure | Load model, capture activations, execute tools, save results |
| **F1** | Sparse Autoencoders (SAE) | Discover and inspect sparse features |
| **F2** | J-Space | Read out internal representations using the Jacobian Lens |
| **F3** | Agentic Investigation v0 | Agent selects tools, tests hypotheses, produces a report (**Alpha Release Target**) |
| **F4** | Natural Language Autoencoders (NLA) | Translate supported activations into natural-language descriptions |
| **F5** | Block-Sparse Featurizers | Discover multidimensional visual concepts |
| **F6** | Manifold Steering | Intervene along learned representation geometry |
| **F7** | AutoRefine | Propose, apply, and verify controlled model refinements |

---

## 3. Implementation Checklist (12 / 40 Completed)

### F0 — Foundation (5/5) [Completed]
- [x] Create `autoexplain` Python package with installable CLI
- [x] Build a PyTorch model adapter for activations, gradients, and interventions
- [x] Define common tool inputs, outputs, and compatibility metadata
- [x] Save experiment configurations, model identity, random seeds, and results
- [x] Generate a basic HTML investigation report  
*Done when:* One example model runs through a reproducible investigation and produces a report. (Completed with GPT-2 demo and HTML report generation)

### F1 — SAE (0/5)
- [ ] Integrate SAE Lens or another established SAE implementation
- [ ] Load pretrained SAE checkpoints and optionally train a small SAE
- [ ] Extract and visualize top-activating examples
- [ ] Evaluate reconstruction quality and feature sparsity
- [ ] Test feature ablation against random-feature controls  
*Done when:* User can inspect SAE features and measure whether changing them affects model behavior.

### F2 — J-Space (0/5)
- [ ] Integrate the Jacobian Lens for supported language models
- [ ] Compute and cache layerwise Jacobian readouts
- [ ] Display token/concept readouts across layers
- [ ] Compare Jacobian Lens with logit lens and SAE outputs
- [ ] Test selected directions through controlled interventions  
*Done when:* User can inspect layerwise internal readouts with clearly documented limitations.

### F3 — Agentic Investigation (0/5)
- [x] Create an XAI tool registry with supported model types
- [ ] Build hypothesis generation and experiment selection
- [ ] Implement bounded trial-and-error execution with resource budgets
- [x] Maintain supporting and contradicting evidence for every hypothesis
- [x] Generate summaries with plots, limitations, and next experiments  
*Done when:* Agent can investigate one question using at least two tools and generate an evidence-backed report.

### F4 — NLA (4/5) [Implemented]
- [x] Integrate released NLA inference implementations and compatible checkpoints (`kitft/nla-models`)
- [x] Convert supported LLM activations into natural-language descriptions (`ActivationVerbalizer`)
- [x] Run text-to-activation reconstruction and score fidelity (`ActivationReconstructor` / `NLACritic`)
- [ ] Compare NLA descriptions with SAE/J-Space readouts
- [x] Expose reconstruction and semantic limitations in reports (direction-MSE, Cosine Sim, random controls)  
*Done when:* Agent can use natural-language descriptions as testable hypotheses, not automatically verified explanations.

### F5 — Block-Sparse Featurizers (0/5)
- [ ] Integrate Goodfire's published block-sparse implementation
- [ ] Extract patch-level vision representations
- [ ] Train or load block-sparse featurizers
- [ ] Visualize block activity and internal concept coordinates
- [ ] Compare against SAE baselines for reconstruction and concept coherence  
*Done when:* One vision-model demo discovers interpretable multidimensional concepts with measurable advantages or limitations.

### F6 — Manifold Steering (0/5)
- [ ] Build a linear steering baseline
- [ ] Identify candidate manifold coordinates from discovered features
- [ ] Implement geometry-aware activation interventions
- [ ] Compare target effects, collateral changes, and robustness
- [ ] Return steering trajectories and causal experiment reports  
*Done when:* Manifold steering produces a measured behavioral change and is compared fairly with linear steering.

### F7 — AutoRefine (0/5)
- [ ] Propose a bounded model modification from investigation evidence
- [ ] Create a modified model or adapter without overwriting the original
- [ ] Evaluate target behavior and unrelated tasks
- [ ] Verify improvement on held-out data and report regressions
- [ ] Support rollback and side-by-side model comparison  
*Done when:* A proposed refinement produces a reproducible before/after evaluation, including any failed fixes.

---

## 4. Existing Foundations to Build Upon

| Method | Upstream Starting Point | Integration Strategy |
| :--- | :--- | :--- |
| **SAE** | [SAE Lens](https://github.com/jbloomAus/SAELens) | Wrap mature SAE infrastructure for loading checkpoints, calculating loss, and steering activations. |
| **J-Space** | Anthropic Jacobian Lens | Layerwise gradient readouts for decoder LMs. |
| **NLA** | [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) | Checkpoints for supported layers with activation-to-text decoding. |
| **Block-Sparse** | [`goodfire-ai/block-sparse-featurizer`](https://github.com/goodfire-ai/block-sparse-featurizer) | MIT-licensed multidimensional visual concept discovery. |
| **Manifold Steering** | Goodfire research | Non-linear geometric steering versus linear steering baselines. |

*Core Design Rule:* Wrap validated implementations rather than reinventing each paper. AutoExplain provides the common experiment interface, automated reasoning agent, and causal verification loop.

---

## 5. Alpha Release Scope (F0–F3)

- **Target:** 1 supported LM (e.g. GPT-2 small / Pythia-70m / Llama-3-1B), SAE + J-Space, single Agent loop, automated HTML report.
- **Minimal API:**
  ```python
  from autoexplain import Investigator

  agent = Investigator(model="gpt2-small")
  report = agent.investigate(
      question="What internal features influence this behavior?",
      tools=["sae", "jspace"],
      max_experiments=10,
  )
  report.save_html("investigation.html")
  ```
