"""Reusable Recipe: Record interactive screen walkthroughs with Cua Agent Mouse Cursor.

Trigger: "make video demo for each usecase"
Pipeline:
  1. Spawns ephemeral FastAPI / Uvicorn server on dynamic port.
  2. Launches headless Chromium via Playwright with video capture enabled.
  3. Injects official Cua Presence Cursor (from `cua.presence_cursor_art_svg`) + ripple effect.
  4. Executes simulated user actions (smooth mouse movement, hovers, clicks, inputs).
  5. Flushes raw recording and transcodes with FFmpeg into high-definition H.264 MP4.
  6. Saves only MP4 files directly to `recipes/demos/` (gitignored).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cua
from playwright.sync_api import BrowserContext, Page, sync_playwright

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "recipes" / "demos"


class CuaCursorRecorder:
    """Manages Cua agent cursor visualization and animated mouse interactions."""

    def __init__(
        self,
        page: Page,
        cursor_color: str = "#10B981",
        cursor_size: int = 32,
        agent_label: str = "Cua Agent",
    ):
        self.page = page
        self.cursor_color = cursor_color
        self.cursor_size = cursor_size
        self.agent_label = agent_label
        self._injected = False

    def inject(self) -> None:
        """Inject Cua presence cursor overlay into page DOM."""
        raw_svg = cua.presence_cursor_art_svg("arrow", self.cursor_color, self.cursor_size)
        escaped_svg = raw_svg.replace("'", "\\'").replace("\n", "")

        js_code = f"""
        (() => {{
            if (document.getElementById('cua-cursor-overlay')) return;
            
            const container = document.createElement('div');
            container.id = 'cua-cursor-overlay';
            container.style.position = 'fixed';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100vw';
            container.style.height = '100vh';
            container.style.pointerEvents = 'none';
            container.style.zIndex = '999999';
            
            const cursor = document.createElement('div');
            cursor.id = 'cua-agent-cursor';
            cursor.style.position = 'absolute';
            cursor.style.left = '100px';
            cursor.style.top = '100px';
            cursor.style.transform = 'translate(0, 0)';
            cursor.style.transition = 'left 0.42s cubic-bezier(0.22, 1, 0.36, 1), top 0.42s cubic-bezier(0.22, 1, 0.36, 1)';
            cursor.style.filter = 'drop-shadow(0 4px 8px rgba(0,0,0,0.25))';
            cursor.innerHTML = `
                {escaped_svg}
                <div style="position: absolute; top: 22px; left: 18px; background: {self.cursor_color}; color: #FFFFFF; font-size: 11px; font-weight: 800; font-family: -apple-system, sans-serif; padding: 2px 8px; border-radius: 9999px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); white-space: nowrap; border: 1px solid rgba(255,255,255,0.8);">
                    {self.agent_label}
                </div>
            `;
            
            container.appendChild(cursor);
            document.body.appendChild(container);
            
            window.cuaMove = (x, y) => {{
                cursor.style.left = `${{x}}px`;
                cursor.style.top = `${{y}}px`;
            }};
            
            window.cuaClickEffect = (x, y) => {{
                const ripple = document.createElement('div');
                ripple.style.position = 'absolute';
                ripple.style.left = `${{x}}px`;
                ripple.style.top = `${{y}}px`;
                ripple.style.width = '14px';
                ripple.style.height = '14px';
                ripple.style.border = '4px solid #EF4444';
                ripple.style.borderRadius = '50%';
                ripple.style.transform = 'translate(-50%, -50%) scale(1)';
                ripple.style.opacity = '1';
                ripple.style.transition = 'all 0.45s cubic-bezier(0.16, 1, 0.3, 1)';
                container.appendChild(ripple);
                
                requestAnimationFrame(() => {{
                    ripple.style.transform = 'translate(-50%, -50%) scale(4.5)';
                    ripple.style.opacity = '0';
                }});
                setTimeout(() => ripple.remove(), 500);
            }};
        }})();
        """
        self.page.evaluate(js_code)
        self.page.wait_for_timeout(300)
        self._injected = True

    def move_to(self, selector: str, desc: str = "") -> tuple[float, float]:
        """Smoothly moves cursor to target selector."""
        loc = self.page.locator(selector).first
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        if not box:
            raise ValueError(f"Could not calculate bounding box for: {selector}")

        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        print(f"  [Cua Mouse] Move to {desc or selector} -> ({x:.1f}, {y:.1f})")
        self.page.evaluate(f"window.cuaMove({x}, {y})")
        self.page.wait_for_timeout(450)
        return x, y

    def hover(self, selector: str, desc: str = "", duration: float = 1.2) -> None:
        """Moves cursor to element, hovers to trigger CSS states, and waits."""
        x, y = self.move_to(selector, desc)
        self.page.locator(selector).first.hover()
        self.page.wait_for_timeout(int(duration * 1000))

    def click(self, selector: str, desc: str = "", delay_after: float = 1.5) -> None:
        """Moves cursor, triggers click ripple, and dispatches native click."""
        x, y = self.move_to(selector, desc)
        self.page.evaluate(f"window.cuaClickEffect({x}, {y})")
        self.page.wait_for_timeout(100)
        self.page.locator(selector).first.click()
        self.page.wait_for_timeout(int(delay_after * 1000))

    def type_text(self, selector: str, text: str, desc: str = "", delay_after: float = 1.0) -> None:
        """Clicks input and types text."""
        self.click(selector, desc=f"Input: {desc or selector}", delay_after=0.3)
        self.page.locator(selector).first.fill(text)
        self.page.wait_for_timeout(int(delay_after * 1000))


def find_free_port() -> int:
    """Find available ephemeral localhost port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Wait until server responds to health endpoint."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def transcode_to_mp4(raw_webm: Path, output_base: Path) -> Path:
    """Transcodes raw Playwright WebM into high-quality MP4 and cleans up raw files."""
    mp4_path = output_base.with_suffix(".mp4")

    print(f"Transcoding {raw_webm.name} -> {mp4_path.name}...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_webm),
            "-c:v", "libx264", "-preset", "slow", "-crf", "22",
            "-pix_fmt", "yuv420p",
            str(mp4_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp4_path


# ==============================================================================
# The 5 Core Use Case Scenarios
# ==============================================================================

def scenario_1_zero_setup_arrival(page: Page, cua: CuaCursorRecorder) -> None:
    """Use Case 1: Zero-Setup Arrival & Exploration.
    User lands on AutoExplain Studio with pre-loaded sentence 'The capital of France is'
    and token activation strength tints under Unsloth Light Theme.
    """
    cua.hover(".brand", desc="AutoExplain Studio Brand Header", duration=1.0)
    cua.hover(".model-pills-group", desc="Model Switcher Pill Group", duration=1.0)
    cua.hover(".prompt-input", desc="Active Prompt Input Bar", duration=1.2)

    # Hover over each word pill in sequence to inspect activation tints
    cua.hover(".token-pill:nth-child(1)", desc="Word Pill 'The' (12% strength)", duration=1.2)
    cua.hover(".token-pill:nth-child(2)", desc="Word Pill 'capital' (88% strength)", duration=1.2)
    cua.hover(".token-pill:nth-child(3)", desc="Word Pill 'of' (25% strength)", duration=1.2)
    cua.hover(".token-pill:nth-child(4)", desc="Word Pill 'France' (95% strength)", duration=1.2)
    cua.hover(".token-pill:nth-child(5)", desc="Word Pill 'is' (99% strength)", duration=1.5)

    # Glance over the two cards
    cua.hover("#interpretCard .card-header", desc="Card 1: 1. Interpret", duration=1.2)
    cua.hover("#steerCard .card-header", desc="Card 2: 2. Steer", duration=1.5)


def scenario_2_word_by_word_interpreter(page: Page, cua: CuaCursorRecorder) -> None:
    """Use Case 2: The Interpreter — Word-by-Word Concept Firing.
    User clicks through ALL 5 words systematically: 'The', 'capital', 'of', 'France', 'is'.
    Card 1 dynamically updates with exact concept readouts and activation tags.
    """
    # 1. Click "The" (Pos 0) -> Syntax / Determiner
    cua.click(".token-pill:nth-child(1)", desc="Word Pill 'The' [Pos 0]", delay_after=2.2)

    # 2. Click "capital" (Pos 1) -> Administrative seat / Governance
    cua.click(".token-pill:nth-child(2)", desc="Word Pill 'capital' [Pos 1]", delay_after=2.2)

    # 3. Click "of" (Pos 2) -> Relational preposition
    cua.click(".token-pill:nth-child(3)", desc="Word Pill 'of' [Pos 2]", delay_after=2.2)

    # 4. Click "France" (Pos 3) -> French Republic Sovereign entity
    cua.click(".token-pill:nth-child(4)", desc="Word Pill 'France' [Pos 3]", delay_after=2.2)

    # 5. Click "is" (Pos 4) -> Predicting 'Paris' as capital
    cua.click(".token-pill:nth-child(5)", desc="Word Pill 'is' [Pos 4]", delay_after=2.5)


def scenario_3_causal_steering_tokyo(page: Page, cua: CuaCursorRecorder) -> None:
    """Use Case 3: Causal Steering — Entity Intervention (Tokyo Steer).
    User copies concept into Card 2, selects '📌 Tokyo Steer', applies steer,
    and observes side-by-side diff between Default and Steered outputs.
    """
    # Click "France" token to inspect sovereign entity
    cua.click(".token-pill:nth-child(4)", desc="Word Pill 'France'", delay_after=1.8)

    # Transfer concept into Card 2
    cua.click(".btn-quick-steer", desc="Button '⚡ Steer This Concept in Card 2'", delay_after=2.0)

    # Select preset "📌 Tokyo Steer"
    cua.click(".preset-pill:nth-child(1)", desc="Preset '📌 Tokyo Steer'", delay_after=2.0)

    # Apply steer
    cua.click(".btn-primary", desc="Button [ Apply Steer ]", delay_after=2.8)

    # Hover over side-by-side output comparison
    cua.hover(".comparison-container", desc="Side-by-Side Diff: Paris -> Tokyo", duration=2.5)


def scenario_4_suppress_thought_and_reset(page: Page, cua: CuaCursorRecorder) -> None:
    """Use Case 4: Concept Suppression & Baseline Recovery.
    User selects '🚫 Suppress Paris', observes fallback to generic descriptions,
    and clicks 'Reset Baseline' to restore default model behavior.
    """
    # Select preset "🚫 Suppress Paris"
    cua.click(".preset-pill:nth-child(3)", desc="Preset '🚫 Suppress Paris'", delay_after=2.2)

    # Apply steer intervention
    cua.click(".btn-primary", desc="Button [ Apply Steer ]", delay_after=2.8)

    # Hover over diff banner
    cua.hover("#steerDiffBanner", desc="Diff Banner (Suppression of Paris)", duration=2.0)

    # Click Reset Baseline
    cua.click("#steerCard .card-header .btn-secondary", desc="Button [ Reset Baseline ]", delay_after=2.5)

    # Verify baseline restored
    cua.hover(".comparison-container", desc="Baseline Model Output Restored", duration=2.0)


def scenario_5_cross_model_switch(page: Page, cua: CuaCursorRecorder) -> None:
    """Use Case 5: Cross-Model Switch & Checkpoint Upload.
    User saves current steer preset, switches active model to 'gpt2-finetuned.pt'
    without reloading, and uploads a custom checkpoint.
    """
    # Apply Tokyo steer first
    cua.click(".preset-pill:nth-child(1)", desc="Preset '📌 Tokyo Steer'", delay_after=1.8)

    # Save steer preset
    page.on("dialog", lambda dialog: dialog.accept("📌 Tokyo Alternate"))
    cua.click(".steer-action-buttons .btn-secondary", desc="Button [ Save Steer ]", delay_after=2.2)

    # Switch to finetuned model
    cua.click(".model-pill:nth-child(2)", desc="Switch to 'gpt2-finetuned.pt'", delay_after=2.5)

    # Upload custom checkpoint
    cua.click("#uploadBtn", desc="Button [ + Upload .pt ]", delay_after=1.0)
    page.set_input_files("#checkpointFileInput", {
        "name": "custom_finetune.pt",
        "mimeType": "application/octet-stream",
        "buffer": b"checkpoint-dummy-binary-data"
    })
    page.wait_for_timeout(2500)
    cua.hover(".model-pills-group", desc="Model Switcher with Uploaded Checkpoint", duration=2.0)


def default_full_walkthrough_scenario(page: Page, cua: CuaCursorRecorder) -> None:
    """Comprehensive walkthrough covering all 5 core stories in one continuous demo."""
    # Step 1: Click word pill "France"
    cua.click(".token-pill:nth-child(4)", desc="Word Pill 'France'", delay_after=2.0)

    # Step 2: Transfer concept to Card 2
    cua.click(".btn-quick-steer", desc="Button '⚡ Steer This Concept'", delay_after=2.0)

    # Step 3: Inspect Prediction Node "is"
    cua.click(".token-pill:nth-child(5)", desc="Word Pill 'is' (Capital Prediction)", delay_after=2.0)

    # Step 4: Click preset "📌 Tokyo Steer"
    cua.click(".preset-pill:nth-child(1)", desc="Preset '📌 Tokyo Steer'", delay_after=2.0)

    # Step 5: Execute steer
    cua.click(".btn-primary", desc="Primary Button [ Apply Steer ]", delay_after=2.5)

    # Step 6: Negative steering
    cua.click(".preset-pill:nth-child(3)", desc="Preset '🚫 Suppress Paris'", delay_after=2.5)

    # Step 7: Save preset (handle alert dialog)
    page.on("dialog", lambda dialog: dialog.accept("📌 Paris Suppressed"))
    cua.click(".steer-action-buttons .btn-secondary", desc="Button [ Save Steer ]", delay_after=2.0)

    # Step 8: Cross-model switch
    cua.click(".model-pill:nth-child(2)", desc="Model Switcher 'gpt2-finetuned.pt'", delay_after=2.5)

    # Step 9: Reset baseline
    cua.click("#steerCard .card-header .btn-secondary", desc="Button [ Reset Baseline ]", delay_after=2.5)


USECASE_REGISTRY: Dict[str, Tuple[str, Callable[[Page, CuaCursorRecorder], None], str]] = {
    "1": ("usecase_1_zero_setup_arrival", scenario_1_zero_setup_arrival, "Zero-Setup Arrival & Exploration"),
    "2": ("usecase_2_word_by_word_interpreter", scenario_2_word_by_word_interpreter, "Word-by-Word Concept Interpreter"),
    "3": ("usecase_3_causal_steering_tokyo", scenario_3_causal_steering_tokyo, "Causal Steering (Tokyo Steer)"),
    "4": ("usecase_4_suppress_thought_and_reset", scenario_4_suppress_thought_and_reset, "Concept Suppression & Baseline Recovery"),
    "5": ("usecase_5_cross_model_switch", scenario_5_cross_model_switch, "Cross-Model Switch & Checkpoint Upload"),
}


def run_recording_pipeline(
    scenario_fn: Callable[[Page, CuaCursorRecorder], None],
    output_filename: str = "AutoExplain_Interactive_Screen_Demo",
    output_dir: Optional[Path] = None,
) -> Path:
    """End-to-end pipeline executing a single scenario and producing MP4 video."""
    target_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_video_dir = target_dir / ".temp_video_recordings"
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    print(f"Starting ephemeral AutoExplain server on port {port}...")
    server_proc = subprocess.Popen(
        ["uvicorn", "autoexplain.web.app:app", "--host", "127.0.0.1", f"--port={port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        url = f"http://127.0.0.1:{port}"
        if not wait_for_server(url):
            raise RuntimeError(f"Server on {url} failed to respond within timeout.")

        print("Launching Playwright with video capture...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 840},
                device_scale_factor=2,
                record_video_dir=str(temp_video_dir),
                record_video_size={"width": 1280, "height": 840},
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(800)

            recorder = CuaCursorRecorder(page)
            recorder.inject()

            print(f"Running interactive scenario '{output_filename}'...")
            scenario_fn(page, recorder)

            page.wait_for_timeout(1000)
            page.close()
            context.close()
            browser.close()

        recorded = list(temp_video_dir.glob("*.webm"))
        if not recorded:
            raise RuntimeError("Playwright recorded video not found.")
        raw_video = recorded[0]

        output_base = target_dir / output_filename
        mp4_path = transcode_to_mp4(raw_video, output_base)

        shutil.rmtree(temp_video_dir, ignore_errors=True)
        return mp4_path

    finally:
        server_proc.terminate()
        server_proc.wait()


def record_all_usecases(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Records all 5 distinct use-case scenarios sequentially in one server session."""
    target_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_video_dir = target_dir / ".temp_video_recordings"
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    print(f"Starting ephemeral AutoExplain server on port {port}...")
    server_proc = subprocess.Popen(
        ["uvicorn", "autoexplain.web.app:app", "--host", "127.0.0.1", f"--port={port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    results: Dict[str, Path] = {}
    try:
        url = f"http://127.0.0.1:{port}"
        if not wait_for_server(url):
            raise RuntimeError(f"Server on {url} failed to respond within timeout.")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            for key, (filename, scenario_fn, title) in USECASE_REGISTRY.items():
                print(f"\n========================================================")
                print(f"🎥 Recording Use Case {key}: {title}")
                print(f"========================================================")

                scenario_temp_dir = temp_video_dir / f"scenario_{key}"
                scenario_temp_dir.mkdir(parents=True, exist_ok=True)

                context = browser.new_context(
                    viewport={"width": 1280, "height": 840},
                    device_scale_factor=2,
                    record_video_dir=str(scenario_temp_dir),
                    record_video_size={"width": 1280, "height": 840},
                )
                page = context.new_page()

                # Reset server session to clean baseline before each usecase
                try:
                    req = urllib.request.Request(
                        f"{url}/api/reset",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=2)
                except Exception:
                    pass

                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(800)

                recorder = CuaCursorRecorder(page)
                recorder.inject()

                scenario_fn(page, recorder)

                page.wait_for_timeout(1000)
                page.close()
                context.close()

                recorded = list(scenario_temp_dir.glob("*.webm"))
                if not recorded:
                    raise RuntimeError(f"Recording for use case {key} failed.")
                raw_video = recorded[0]

                output_base = target_dir / filename
                mp4_path = transcode_to_mp4(raw_video, output_base)
                results[f"Usecase {key}: {title}"] = mp4_path
                print(f"  ✓ Saved: {mp4_path.name} ({mp4_path.stat().st_size / (1024*1024):.2f} MB)")

            browser.close()

        shutil.rmtree(temp_video_dir, ignore_errors=True)
        return results

    finally:
        server_proc.terminate()
        server_proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record interactive screen demo with Cua mouse cursor (MP4 only).")
    parser.add_argument("--usecase", choices=["1", "2", "3", "4", "5", "all", "full"], default="all",
                        help="Which usecase to record ('1'-'5', 'all' for all 5 usecases, or 'full' for continuous walkthrough).")
    parser.add_argument("--name", default="", help="Custom output filename (optional).")
    args = parser.parse_args()

    if args.usecase == "all":
        print("Recording all 5 usecase video demos into recipes/demos/...")
        all_results = record_all_usecases()
        print("\n🎉 Completed all 5 usecase video recordings:")
        for label, path in all_results.items():
            print(f"  [{label}] -> {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
    elif args.usecase == "full":
        out_name = args.name or "AutoExplain_Interactive_Screen_Demo"
        mp4_file = run_recording_pipeline(
            scenario_fn=default_full_walkthrough_scenario,
            output_filename=out_name,
        )
        print(f"\n[MP4] {mp4_file} ({mp4_file.stat().st_size / (1024*1024):.2f} MB)")
    else:
        filename, fn, title = USECASE_REGISTRY[args.usecase]
        out_name = args.name or filename
        mp4_file = run_recording_pipeline(
            scenario_fn=fn,
            output_filename=out_name,
        )
        print(f"\n[MP4] {mp4_file} ({mp4_file.stat().st_size / (1024*1024):.2f} MB)")
