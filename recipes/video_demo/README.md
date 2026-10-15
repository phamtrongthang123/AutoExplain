# Video Demo Recipe: Interactive Screen with Cua Agent Mouse Cursor

This recipe defines the standardized procedure for generating interactive screen recording videos with visible, animated agent mouse telemetry whenever the command **`"make video demo for each usecase"`** is requested.

---

## 🎯 Trigger Phrase

> **`"make video demo for each usecase"`**

Whenever this command is given:
1. Spin up an ephemeral backend server (`uvicorn autoexplain.web.app:app` on an available port).
2. Launch headless Chromium via Playwright with video capture enabled (`record_video_dir`).
3. Inject the official **Cua Presence Cursor** (`cua.presence_cursor_art_svg`) into the DOM overlay with easing trajectory physics and expanding click ripples.
4. Execute the targeted scenario step-by-step with realistic human/agent pacing (smooth movements, hover states, input typing, and click actions).
5. Transcode the raw Chromium video stream into:
   - **`*.mp4`** only (H.264 standard video with YUV420P encoding for universal playback).
6. Present the MP4 output file with clickable links.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AutoExplain UI                        │
│             FastAPI Backend + Unsloth Light UI              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   Playwright Headless Chromium     Cua Presence Cursor Engine
   • 1280x840 Retina viewport       • Official Cua SVG cursor
   • Raw frame recording (.webm)    • Bezier easing curves
   • Real DOM event dispatch        • Expanding red click ripples
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                        FFmpeg Pipeline
   • Libx264 H.264 MP4 (CRF 22, YUV420P) — Single cleanly exported MP4
```

---

## 📋 The 4 Core Use Cases

| # | Use Case | Scenario Actions | Target Element & Behavior |
| :---: | :--- | :--- | :--- |
| **1** | **Zero-Setup Arrival** | Cursor enters, scrolls prompt row, hovers over word pills. | Displays prompt `"The capital of France is"` with token activation tints under Unsloth Light Theme. |
| **2** | **The Interpreter** | Clicks `"France"` $\to$ reads Card 1 $\to$ clicks `"is"`. | Card 1 displays plain-English firing concepts without raw neuron coordinates or histograms. |
| **3** | **The Steerer** | Clicks `⚡ Steer This Concept` $\to$ clicks `📌 Tokyo Steer` $\to$ clicks `[ Apply Steer ]`. | Card 2 displays side-by-side Default vs. Steered output with highlighted token differences. |
| **4** | **Cross-Model Switch** | Clicks `🚫 Suppress Paris` $\to$ clicks `Save Steer` $\to$ clicks `gpt2-finetuned.pt`. | Preserves saved steer presets across model checkpoints in the top header without reloading. |

---

## 🚀 Usage

### 1. Run Complete Default Walkthrough
```bash
python recipes/video_demo/record_demo.py --name AutoExplain_Interactive_Screen_Demo
# Output saved to: recipes/demos/AutoExplain_Interactive_Screen_Demo.mp4 (gitignored)
```

### 2. Run for a Specific Use Case
To record an individual usecase scenario, define a Python scenario function and invoke `run_recording_pipeline`:

```python
from recipes.video_demo.record_demo import run_recording_pipeline, CuaCursorRecorder
from playwright.sync_api import Page

def usecase_interpret_scenario(page: Page, cua: CuaCursorRecorder):
    # Step 1: Click word pill "France"
    cua.click(".token-pill:nth-child(4)", desc="Word Pill 'France'", delay_after=2.0)
    # Step 2: Transfer concept to Card 2
    cua.click(".btn-quick-steer", desc="Steer Concept Button", delay_after=2.0)

# Output saved to recipes/demos/usecase_2_interpreter_demo.mp4
mp4_path = run_recording_pipeline(
    scenario_fn=usecase_interpret_scenario,
    output_filename="usecase_2_interpreter_demo",
)
```

---

## 📦 Required Dependencies
All dependencies are installed in the workspace environment:
- `cua` & `cua-driver` (Cua cursor art & presence protocols)
- `playwright` (`pip install playwright` + `playwright install chromium`)
- `ffmpeg` (system package for video transcoding)
- `uvicorn` & `fastapi` (web application backend)
