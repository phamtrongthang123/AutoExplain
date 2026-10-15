"""Reusable Recipe: Record interactive screen walkthroughs with Cua Agent Mouse Cursor.

Trigger: "make video demo for each usecase"
Pipeline:
  1. Spawns ephemeral FastAPI / Uvicorn server on dynamic port.
  2. Launches headless Chromium via Playwright with video capture enabled.
  3. Injects official Cua Presence Cursor (from `cua.presence_cursor_art_svg`) + ripple effect.
  4. Executes simulated user actions (smooth mouse movement, clicks, inputs).
  5. Flushes raw recording and transcodes with FFmpeg into high-quality MP4, WebM, and animated GIF.
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
from typing import Callable, Optional

import cua
from playwright.sync_api import BrowserContext, Page, sync_playwright

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT


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


def transcode_video(raw_webm: Path, output_base: Path, generate_gif: bool = True) -> dict[str, Path]:
    """Transcodes raw Playwright WebM into MP4, WebM, and animated GIF."""
    mp4_path = output_base.with_suffix(".mp4")
    webm_path = output_base.with_suffix(".webm")
    gif_path = output_base.with_suffix(".gif")

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

    shutil.copyfile(raw_webm, webm_path)
    results = {"mp4": mp4_path, "webm": webm_path}

    if generate_gif:
        print(f"Generating optimized GIF: {gif_path.name}...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4_path),
                "-vf", "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(gif_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        results["gif"] = gif_path

    return results


def run_recording_pipeline(
    scenario_fn: Callable[[Page, CuaCursorRecorder], None],
    output_filename: str = "AutoExplain_Interactive_Screen_Demo",
    output_dir: Optional[Path] = None,
    generate_gif: bool = True,
) -> dict[str, Path]:
    """End-to-end pipeline executing a scenario and producing video assets."""
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

            # Initialize Cua cursor
            recorder = CuaCursorRecorder(page)
            recorder.inject()

            # Execute scenario
            print("Running interactive scenario...")
            scenario_fn(page, recorder)

            page.wait_for_timeout(1000)
            page.close()
            context.close()
            browser.close()

        # Find output file
        recorded = list(temp_video_dir.glob("*.webm"))
        if not recorded:
            raise RuntimeError("Playwright recorded video not found.")
        raw_video = recorded[0]

        output_base = target_dir / output_filename
        artifacts = transcode_video(raw_video, output_base, generate_gif=generate_gif)

        # Cleanup temp directory
        shutil.rmtree(temp_video_dir, ignore_errors=True)
        return artifacts

    finally:
        server_proc.terminate()
        server_proc.wait()


def default_full_walkthrough_scenario(page: Page, cua_rec: CuaCursorRecorder) -> None:
    """Default end-to-end user walkthrough covering all 4 core stories."""
    # Step 1: Click word pill "France"
    cua_rec.click(".token-pill:nth-child(4)", desc="Word Pill 'France'", delay_after=2.0)

    # Step 2: Transfer concept to Card 2
    cua_rec.click(".btn-quick-steer", desc="Button '⚡ Steer This Concept'", delay_after=2.0)

    # Step 3: Inspect Prediction Node "is"
    cua_rec.click(".token-pill:nth-child(5)", desc="Word Pill 'is' (Capital Prediction)", delay_after=2.0)

    # Step 4: Click preset "📌 Tokyo Steer"
    cua_rec.click(".preset-pill:nth-child(1)", desc="Preset '📌 Tokyo Steer'", delay_after=2.0)

    # Step 5: Execute steer
    cua_rec.click(".btn-primary", desc="Primary Button [ Apply Steer ]", delay_after=2.5)

    # Step 6: Negative steering
    cua_rec.click(".preset-pill:nth-child(3)", desc="Preset '🚫 Suppress Paris'", delay_after=2.5)

    # Step 7: Save preset (handle alert dialog)
    page.on("dialog", lambda dialog: dialog.accept("📌 Paris Suppressed"))
    cua_rec.click(".steer-action-buttons .btn-secondary", desc="Button [ Save Steer ]", delay_after=2.0)

    # Step 8: Cross-model switch
    cua_rec.click(".model-pill:nth-child(2)", desc="Model Switcher 'gpt2-finetuned.pt'", delay_after=2.5)

    # Step 9: Reset baseline
    cua_rec.click("#steerCard .card-header .btn-secondary", desc="Button [ Reset Baseline ]", delay_after=2.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record interactive screen demo with Cua mouse cursor.")
    parser.add_argument("--name", default="AutoExplain_Interactive_Screen_Demo", help="Output filename base.")
    parser.add_argument("--no-gif", action="store_true", help="Skip animated GIF generation.")
    args = parser.parse_args()

    results = run_recording_pipeline(
        scenario_fn=default_full_walkthrough_scenario,
        output_filename=args.name,
        generate_gif=not args.no_gif,
    )
    print("\nRecorded files:")
    for k, v in results.items():
        print(f"  [{k.upper()}] {v} ({v.stat().st_size / (1024*1024):.2f} MB)")
