"""FastAPI web application for AutoExplain UIUX (Unsloth Studio Light Theme)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = BASE_DIR / "fixtures" / "neuronpedia_mock.json"
STATIC_DIR = BASE_DIR / "static"


class SelectModelRequest(BaseModel):
    model: str


class InterpretRequest(BaseModel):
    prompt: Optional[str] = None
    token_index: Optional[int] = None


class SteerRequest(BaseModel):
    prompt: Optional[str] = None
    concept: str
    intensity: float = 1.0
    mode: str = "boost"  # "boost" or "suppress"


class SavePresetRequest(BaseModel):
    name: str
    concept: str
    intensity: float = 1.0
    mode: str = "boost"


def load_fixture() -> Dict[str, Any]:
    """Load mock data fixture."""
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Fixture not found at {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    """Factory creating and configuring the AutoExplain web application."""
    app = FastAPI(
        title="AutoExplain Studio",
        description="Minimalist 2-Card Mechanistic Interpretability & Steering Interface",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize in-memory session state from fixture
    fixture = load_fixture()
    state: Dict[str, Any] = {
        "models": list(fixture["models"]),
        "active_model": fixture["default_model"],
        "default_prompt": fixture["default_prompt"],
        "active_prompt": fixture["default_prompt"],
        "prompts_data": dict(fixture["prompts"]),
        "presets": list(fixture["initial_presets"]),
        "active_steer": None,
        "selected_token_index": 4,  # default to "is"
    }

    # Helper: resolve prompt data
    def get_prompt_info(prompt_text: str) -> Dict[str, Any]:
        data = state["prompts_data"].get(prompt_text)
        if data:
            return data
        # Synthesize fallback tokenization for novel prompts
        words = prompt_text.strip().split()
        tokens = [
            {
                "index": i,
                "text": w,
                "activation_score": 0.5 + (0.1 * (i % 5)),
                "concept": f"Syntactic & semantic token representation for '{w}'",
                "tags": ["General Representation"],
                "confidence": 0.85,
            }
            for i, w in enumerate(words)
        ]
        return {
            "text": prompt_text,
            "default_output": f"{prompt_text} ...",
            "tokens": tokens,
            "steer_scenarios": {},
        }

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>AutoExplain UI is building...</h1>")
        return FileResponse(index_file)

    @app.get("/api/status")
    async def get_status():
        prompt_data = get_prompt_info(state["active_prompt"])
        idx = state["selected_token_index"]
        tokens = prompt_data["tokens"]
        selected_token = tokens[idx] if (0 <= idx < len(tokens)) else tokens[-1]

        default_output = prompt_data["default_output"]
        steered_output = default_output
        if state["active_steer"]:
            concept_name = state["active_steer"]["concept"]
            scenario = prompt_data.get("steer_scenarios", {}).get(concept_name)
            if scenario:
                steered_output = scenario["steered_output"]
            else:
                steered_output = f"{prompt_data['text']} [steered toward '{concept_name}']"

        return {
            "active_model": state["active_model"],
            "models": state["models"],
            "prompt": state["active_prompt"],
            "tokens": tokens,
            "selected_token_index": idx,
            "selected_token": selected_token,
            "default_output": default_output,
            "steered_output": steered_output,
            "active_steer": state["active_steer"],
            "presets": state["presets"],
        }

    @app.get("/api/models")
    async def list_models():
        return {
            "active_model": state["active_model"],
            "models": state["models"],
        }

    @app.post("/api/models/select")
    async def select_model(req: SelectModelRequest):
        model_ids = [m["id"] for m in state["models"]]
        if req.model not in model_ids:
            # Register dynamically if new
            state["models"].append({
                "id": req.model,
                "name": req.model,
                "size": "Custom",
                "status": "ready",
            })
        state["active_model"] = req.model
        return {"status": "ok", "active_model": state["active_model"]}

    @app.post("/api/models/upload")
    async def upload_model_checkpoint(file: UploadFile = File(...)):
        filename = file.filename or "uploaded_checkpoint.pt"
        if not filename.endswith(".pt") and not filename.endswith(".pth"):
            filename = f"{filename}.pt"

        # Register uploaded model placeholder
        new_model = {
            "id": filename,
            "name": f"Uploaded ({filename})",
            "size": "Custom",
            "status": "ready",
        }
        if not any(m["id"] == filename for m in state["models"]):
            state["models"].append(new_model)
        state["active_model"] = filename

        return {
            "status": "ok",
            "message": f"Checkpoint {filename} loaded successfully.",
            "active_model": filename,
        }

    @app.post("/api/interpret")
    async def interpret_token(req: InterpretRequest):
        prompt_text = req.prompt or state["active_prompt"]
        prompt_data = get_prompt_info(prompt_text)
        tokens = prompt_data["tokens"]

        if req.token_index is not None:
            if req.token_index < 0 or req.token_index >= len(tokens):
                raise HTTPException(status_code=400, detail="Invalid token index")
            target_idx = req.token_index
        else:
            prev_idx = state.get("selected_token_index", len(tokens) - 1)
            target_idx = min(prev_idx, len(tokens) - 1) if tokens else 0

        state["active_prompt"] = prompt_text
        state["selected_token_index"] = target_idx
        token_info = tokens[target_idx]

        return {
            "token": token_info["text"],
            "index": token_info["index"],
            "concept": token_info["concept"],
            "tags": token_info.get("tags", []),
            "activation_score": token_info.get("activation_score", 0.8),
            "confidence": token_info.get("confidence", 0.95),
            "prompt": prompt_text,
        }

    @app.post("/api/steer")
    async def apply_steer(req: SteerRequest):
        prompt_text = req.prompt or state["active_prompt"]
        prompt_data = get_prompt_info(prompt_text)
        default_out = prompt_data["default_output"]

        # Look up preset or synthesize realistic steered output
        scenarios = prompt_data.get("steer_scenarios", {})
        scenario = scenarios.get(req.concept)

        if scenario:
            steered_out = scenario["steered_output"]
            diff = scenario["difference_summary"]
            thought = scenario["active_thought"]
        elif req.mode == "suppress":
            steered_out = f"{prompt_text} a distinct entity with {req.concept.lower()} traits removed."
            diff = f"Suppressed internal firing for '{req.concept}' with intensity {req.intensity:.2f}."
            thought = f"Suppressed: {req.concept}"
        else:
            steered_out = f"{prompt_text} {req.concept.split('/')[0].strip()}."
            diff = f"Shifted output toward '{req.concept}' representation with intensity {req.intensity:.2f}."
            thought = f"Injected: {req.concept}"

        steer_state = {
            "concept": req.concept,
            "intensity": req.intensity,
            "mode": req.mode,
            "difference_summary": diff,
            "active_thought": thought,
        }
        state["active_steer"] = steer_state

        return {
            "default_output": default_out,
            "steered_output": steered_out,
            "concept": req.concept,
            "intensity": req.intensity,
            "mode": req.mode,
            "difference_summary": diff,
            "active_thought": thought,
            "active_model": state["active_model"],
        }

    @app.get("/api/presets")
    async def list_presets():
        return state["presets"]

    @app.post("/api/presets")
    async def save_preset(req: SavePresetRequest):
        preset_id = f"preset-{uuid4().hex[:8]}"
        preset_name = req.name if req.name.startswith("📌") or req.name.startswith("🚫") else f"📌 {req.name}"
        new_preset = {
            "id": preset_id,
            "name": preset_name,
            "concept": req.concept,
            "intensity": req.intensity,
            "mode": req.mode,
        }
        state["presets"].append(new_preset)
        return {"status": "ok", "preset": new_preset, "presets": state["presets"]}

    @app.delete("/api/presets/{preset_id}")
    async def delete_preset(preset_id: str):
        state["presets"] = [p for p in state["presets"] if p["id"] != preset_id]
        return {"status": "ok", "presets": state["presets"]}

    @app.post("/api/reset")
    async def reset_session():
        state["active_steer"] = None
        state["active_prompt"] = state["default_prompt"]
        state["selected_token_index"] = 4
        return {"status": "ok", "message": "Baseline restored."}

    # Ensure static directory exists and mount static assets
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
