# Scientific Roadmap

AutoExplain is being built to support a long-term research program: understanding what AI models learn, what their predictions leave unexplained, and when interventions improve control without sacrificing useful behavior or human capability.

The questions below define the scientific destination of the project. The five- and ten-year horizons organize research ambitions; they are not promises that a software release will answer these questions on a fixed schedule.

## Research questions

| Actual category | Five-year question 1 | Five-year question 2 | Single ten-year question |
| --- | --- | --- | --- |
| **A — General problem-solving / computation** | **When do accurate neural surrogates fail to reproduce a controllable source system's responses to intervention?** | **Under what conditions do XAI-based internal interventions recover that missing control more selectively than input/output or conventional controls, while retaining a speed advantage?** | **Why do prediction-equivalent neural surrogates differ in the causal control they preserve?** |
| **B — Knowledge and specialization** | **What task rules or constraints can be recovered from expert corrections and interaction traces beyond matched labels and verbal explanations?** | **Under what conditions does that recovered knowledge support correct decisions across experts and task variants, rather than encode one person's habits?** | **Why does task structure make expert knowledge recoverable and transferable through interaction in some cases, but context-bound in others?** |
| **C — Learning and adaptation** | **Which properties of feedback, task context, and the starting model predict whether a correction stays task-specific or generalizes to other behaviors?** | **How can we change that generalization selectively—retaining the intended correction and useful transfer while reducing unwanted effects, including after later updates?** | **Why do some learning conditions favor broad abstractions while others favor task-specific corrections?** |
| **D — Integrated agency and interaction** | **When do model warnings and human interaction signals improve decisions about checking, pausing, or resuming a tool-using agent beyond task state and uncertainty alone?** | **How does repeated delegation change people's unaided ability to detect errors and recover control, and which arrangements preserve that ability?** | **Why does delegating to AI strengthen human capability in some coordination arrangements but erode it in others?** |

## How AutoExplain should contribute

The package should make these questions easier to investigate through reproducible explanations, controlled interventions, and runnable examples. Its scope should grow with the research field, covering more model families and evaluation methods when their assumptions and limitations can be made explicit.

The eventual ambition is to investigate any model, including LLMs, diffusion models, CNNs, U-Nets, decision trees, BERT and JEPA-style models, as well as models used in robotics and healthcare. Architecture families and application domains require different interfaces and standards of evidence. Broad support should therefore mean appropriate methods for each model and task—not one explanation method applied indiscriminately.

For **A**, the toolkit should help compare predictive agreement with intervention agreement, and measure control selectivity and computational cost against appropriate baselines. For **B**, it should help study what expert corrections reveal and test whether recovered constraints transfer across people and task variants. For **C**, it should support tracking intended changes, useful transfer and unwanted effects across subsequent updates. For **D**, it should support instrumented interaction studies while recognizing that software traces alone cannot establish changes in people's unaided capabilities.

These are intended research uses, not implemented capabilities. The current package provides initial PyTorch explanation and steering primitives. Answering the roadmap requires controlled experiments, suitable source systems and datasets, and—where human interaction or healthcare is involved—appropriate study design, consent and ethical review.

The goal is to build evidence toward the questions, including results showing when explanations or interventions fail. Supporting more models is useful only if the resulting experiments help distinguish competing explanations of model and human behavior.
