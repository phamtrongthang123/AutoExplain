# AutoExplain — Feature Roadmap v0.1

Let's focus on UX for now. 
Start by creating autoexplain/web/fixtures/neuronpedia_mock.json with the extracted Neuronpedia mock data, and build autoexplain/web/app.py + autoexplain/web/static/index.html (Unsloth Light Theme) wired to that JSON fixture. Make sure pytest tests/test_web_ux.py passes before touching any PyTorch/GPU backend code


Here are the **User Stories**, **User Goals**, **Expected Outcomes**, and **Constraints** for the webapp, stripped down to the essentials.

---

## Core User Stories

1. **The First-Time Visitor (Zero-Setup Exploration):**
* *"As a user opening the app for the first time, I want to immediately click on words in a pre-loaded sentence (`'The capital of France is'`) and see what the model is thinking, so I don't get stuck wondering what to configure first."*


2. **The Interpreter (Reading Internal Meaning):**
* *"As a user inspecting a prompt, I want to click any word and read a plain-English description of the concept firing at that word, without sifting through layer numbers, feature IDs, or histograms."*


3. **The Steerer (Causal Editing):**
* *"As a user testing how the model works, I want to suppress an active thought or type a new thought in plain English (e.g., `'Japan and Tokyo'`) and see the Default vs. Steered output side by side."*


4. **The Model Owner (Custom `.pt` Checkpoint):**
* *"As a user with my own `.pt` weight file, I want to drop my file into the app and run the exact same Interpret and Steer workflow on my model."* For now we dont need this feature, just do UX first. Btu we still need placeholder.



---

## 1. User Goals (What the user wants to accomplish)

* **Understand *why* a model outputs a specific answer:** See which words in a prompt trigger strong internal activations and what those activations mean in plain human language.
* **Test *"What if?"* interventions effortlessly:** Modify what the model is thinking mid-sentence and observe whether the model's final text actually changes as expected.
* **Avoid tool overwhelm:** Get straight to insight and steering without learning mechanistic interpretability jargon ( SAE dictionary sizes, cosine similarity, or logit tables).

---

## 2. Expected Outcomes (What success looks like)

| User Action | Expected Outcome in the UI |
| --- | --- |
| **Opens the webapp URL** | The page loads directly into a working example (`"The capital of France is"`) with token highlights already visible—no login, no setup wizard, no blank screen. |
| **Clicks a word pill (e.g., `"is"` or `"France"`)** | **Section 1 (Interpret)** immediately displays the plain-English concept firing at that token (e.g., *"Predicting 'Paris' as the capital city of France"*). |
| **Suppresses a thought or enters a new concept (e.g., `"Tokyo / Japan"`)** | **Section 2 (Steer)** updates the side-by-side comparison box, showing the **Default Model Output** (`"...is Paris"`) next to the **Steered Model Output** (`"...is Tokyo"`). |

---

## 3. Constraints (Rules the design must obey)

* **The "Anti-Neuronpedia" Simplicity Constraint:**
* Maximum of **2 main cards** on the screen (`1. Interpret` and `2. Steer`).
* Zero visible math formulas, activation histograms, top/bottom token logit tables, or raw layer/neuron coordinates (`L8 · N14209`).


* **Cold-Start Constraint (Mock-Data First):**
* The frontend must function completely offline/CPU-only using lightweight extracted Neuronpedia JSON data before any live GPU model or `.pt` file is uploaded.


* **Interaction Constraint:**
* Every user input must be either a **single click** (selecting a word pill, clicking `+ Steer`, dragging a simple `Suppress / Boost` slider) or **plain English text** (typing a prompt or a steering concept).


* **Visual Constraint:**
* Strictly **Unsloth Studio Light Theme** (soft sage-white background `#F7FAF8`, pure white cards `#FFFFFF`, emerald-green interactive highlights `#10B981`, high-contrast dark text `#111815`).

## Backend 

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