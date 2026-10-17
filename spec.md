# AutoExplain — Feature Roadmap v0.1

## Frontend (UX First)
Start by creating `autoexplain/web/fixtures/neuronpedia_mock.json` with the extracted Neuronpedia mock data, and build `autoexplain/web/app.py` + `autoexplain/web/static/index.html` (Unsloth Light Theme) wired to that JSON fixture. Make sure `pytest tests/test_web_ux.py` passes before touching any PyTorch/GPU backend code.

### Core User Stories

1. **The Streaming Explorer (Anytime Streaming Discovery):**
   - AutoExplain operates as an anytime streaming discovery engine—like search autocomplete—surfacing the first actionable concept suggestions within 5 seconds of loading a model rather than blocking on an exhaustive batch pipeline. Every 5 seconds, background exploration streams higher-ranked candidates, refines evidence, and promotes causally verified steering interventions as they become available. Users can interact with, inspect, or apply steering to any current result immediately without ever waiting for the search to finish.

2. **The Interpreter (Reading Internal Meaning):**
   - *"As a user inspecting a prompt, I want to click any word and read a plain-English description of the concept firing at that word, without sifting through layer numbers, feature IDs, or histograms."*

3. **The Steerer (Causal Editing):**
   - *"As a user testing how the model works, I want to suppress an active thought or type a new thought in plain English (e.g., `'Japan and Tokyo'`) and see the Default vs. Steered output side by side."*

4. **The Cross-Model Experimenter (Save Steer & Switch `.pt` Model):**
   - *"As a user with multiple checkpoints, I want to save a working steering vector inside the Steer card, switch to another `.pt` model (via a header switcher / `.pt` upload placeholder), and apply that saved steering vector to see its effect on the new model. Because the user may want to know "Do the same interpretability patterns hold consistently across architectures, model sizes, and training conditions?"."*

5. **The Investigator (Testing Interpretations):**
   - *"As a user investigating a model's internal mechanisms, I want to inspect the original neurons and circuits, form hypotheses about their functions, and test those hypotheses through interventions, so I can distinguish plausible interpretations from causally supported explanations."*

### 6. The Librarian (Navigating Internal Knowledge). This usecase is long because I like it the most.

- *"As a researcher exploring an unfamiliar neural network, I want to ask natural-language questions about where information is represented, how internal features are related, and how the network organizes its representations, so I can navigate and understand the model without already knowing which neurons, layers, or circuits to inspect."*

**Example questions:**
- "Where does the model represent information about cats?"
- "Why do these neurons respond to both cats and cars?"
- "How are related concepts organized across layers?"
- "Which other features are connected to this representation?"

**Expected behavior:**
- Find relevant neurons, features, and circuits.
- Discover shared patterns and relationships between representations.
- Explain findings in plain language, with supporting evidence.
- Distinguish observed correlations from causally validated explanations.
- Acknowledge uncertainty when the internal organization is not yet understood.

**Design principle:** AutoExplain should help users discover how a network is already organized, rather than assuming its representations must first be disentangled or reorganized.

**Reference:** [Neural Networks Are Already Clean — Goodfire](https://already-clean.goodfire.com/)

---

### 1. User Goals (What the user wants to accomplish)

- **Understand *why* a model outputs a specific answer:** See which words in a prompt trigger strong internal activations and what those activations mean in plain human language.
- **Test *"What if?"* interventions effortlessly:** Modify what the model is thinking mid-sentence and observe whether the model's final text actually changes as expected.
- **Save and reuse steering vectors across models:** Save an effective steering intervention as a reusable preset pill and test it after switching/uploading another `.pt` checkpoint.
- **Avoid tool overwhelm:** Get straight to insight and steering without learning mechanistic interpretability jargon (SAE dictionary sizes, cosine similarity, or logit tables).

---

### 2. Expected Outcomes (What success looks like)

| User Action | Expected Outcome in the UI |
| :--- | :--- |
| **Opens the webapp URL** | The page loads directly into a working example (`"The capital of France is"`) with token highlights already visible—no login, no setup wizard, no blank screen. |
| **Clicks a word pill (e.g., `"is"` or `"France"`)** | **Card 1 (`1. Interpret`)** immediately displays the plain-English concept firing at that token (e.g., *"Predicting 'Paris' as the capital city of France"*). |
| **Suppresses a thought or enters a new concept (e.g., `"Tokyo / Japan"`)** | **Card 2 (`2. Steer`)** updates the side-by-side comparison box, showing the **Default Model Output** (`"...is Paris"`) next to the **Steered Model Output** (`"...is Tokyo"`). |
| **Clicks `Save Steer` in Card 2** | Saves the current steering vector/concept as a clickable preset pill inside Card 2 (e.g., `📌 Tokyo Steer`). |
| **Uploads a `.pt` file (placeholder) or switches the model pill in the header** | Updates the active model badge in the top bar (e.g., `gpt2-base.pt` $\to$ `custom_weights.pt`) and lets the user click any saved steer pill to test it on the newly selected model. |

---

### 3. Constraints (Rules the design must obey)

- **The "Anti-Neuronpedia" Simplicity Constraint:**
  - Maximum of **2 main cards** on the screen (`1. Interpret` and `2. Steer`). Saved steering vectors and `.pt` model switching must live as lightweight pills inside Card 2 and the Top Header—never as a 3rd card or complex sidebar.
  - Zero visible math formulas, activation histograms, top/bottom token logit tables, or raw layer/neuron coordinates (`L8 · N14209`).
- **Cold-Start Constraint (Mock-Data First):**
  - The frontend must function completely offline/CPU-only using lightweight extracted Neuronpedia JSON data (plus a working `.pt` upload placeholder) before any live GPU model is wired up.
- **Interaction Constraint:**
  - Every user input must be either a **single click** (selecting a word pill, clicking `+ Steer`, clicking `Save Steer`, dragging a `Suppress / Boost` slider) or **plain English text** (typing a prompt or a steering concept).
- **Visual Constraint:**
  - Strictly **Unsloth Studio Light Theme** (soft sage-white background `#F7FAF8`, pure white cards `#FFFFFF`, emerald-green interactive highlights `#10B981`, high-contrast dark text `#111815`).

---

## Backend

AutoExplain provides eight actions on a loaded model, which map directly to the 2-card UI:

- `open_model` & `inspect` → Top Header (`.pt` upload placeholder & model selector)
- `run` & `observe` → Card 1 (`1. Interpret` prompt execution & token thought readout)
- `intervene`, `evaluate`, `reset`, & `record` → Card 2 (`2. Steer` live comparison, reset, and saving steering vectors across models)
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
│   SAE   │    │ Jacobian  │    │    NLA    │   │Block-Spar.│   │ Manifold  │ ... (Methods)
│ Sparse  │    │  Readouts │    │Activation │   │Multidim.  │   │ Steering  │
│Features │    │           │    │  Language │   │ Concepts  │   │ Intervene │
└─────────┘    └───────────┘    └───────────┘   └───────────┘   └───────────┘
```

---

## AI-Based Heuristic Search & Pre-Flight Estimator

AutoExplain functions as an anytime streaming discovery engine powered by an AI-based heuristic search planner, avoiding blind, expensive brute-force passes across neural architectures.

### 1. Literature Knowledge Base (Paper-to-Rule Engine)
- **Architecture Profiling:** Automatically extracts model geometry ($d_{\text{model}}$, layer count, attention heads, activation functions, residual stream dimensions).
- **Paper-Informed Rules:** Maps architectural patterns to proven interpretability methods based on top-tier literature (e.g., Anthropic, Goodfire, OpenAI, TransformerLens):
  - *Small Residual Streams (e.g., GPT-2 small):* Prioritize mid-to-late residual stream sparse autoencoders (Layers 7–9) for clean entity prediction concepts.
  - *Wide/Deeper LLMs (e.g., Llama-style, Mistral):* Prioritize transcoder representations and block-sparse multidimensional concepts.
  - *Dense Attention Networks:* Use Jacobian readout and attention attribution heads for fast syntax and relational bound identification.

### 2. Pre-Flight "Cost vs. Yield" Estimator
Before the user commits compute resources or waits on deep analysis, AutoExplain computes an instant pre-flight estimate to inform the user before paying tremendous search time:
- **Estimated Runtime:** Compares instant heuristic lookup ($\le 5\text{s}$), targeted verification ($\approx 10\text{--}15\text{s}$), and full dictionary sweeps ($> 10\text{min}$).
- **Resource Footprint:** Estimates VRAM usage and memory requirements to proactively prevent Out-Of-Memory (OOM) failures.
- **Expected Concept Yield & Steerability Score:** Predicts the likelihood of discovering clean, causally steerable concepts without causing perplexity degradation or collateral damage.

### 3. Progressive Anytime Priority Scheduling
Execution is prioritized by the ratio of $\frac{\text{Expected Yield}}{\text{Compute Cost}}$:
- **0s — Model Profiling & Triage:** Architectural rules match the checkpoint and initialize the pre-flight estimate card.
- **5s — Fast Heuristic Path:** Surfaces first candidate concepts using lightweight heuristics, cached dictionaries, or sparse projections.
- **10s–15s — Targeted Verification:** Runs minimal causal intervention tests to validate candidate stability.
- **Deep Search (Optional / On-Demand):** Expands to full dictionary training, multidimensional sweeps, or exhaustive causal tracing only when explicitly requested.
