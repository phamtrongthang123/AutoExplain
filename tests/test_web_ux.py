"""Acceptance and UX tests for AutoExplain Web Interface (spec.md)."""

from io import BytesIO
import pytest
from starlette.testclient import TestClient

from autoexplain.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_story_1_first_time_visitor_zero_setup(client):
    """User Story 1: The First-Time Visitor (Zero-Setup Exploration).

    Loads immediately with pre-loaded sentence 'The capital of France is'
    and token highlights without any setup wizard or blank screen.
    """
    # 1. HTML index page loads cleanly
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text
    assert "AutoExplain Studio" in html
    assert "1. Interpret" in html
    assert "2. Steer" in html
    assert "The capital of France is" in html
    assert "#F7FAF8" in html or "--bg: #F7FAF8" in html
    assert "#10B981" in html or "--accent: #10B981" in html

    # 2-Card simplicity: Exactly two main card sections
    assert html.count('class="card"') == 2

    # 2. API Status endpoint provides immediate cold-start state
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["active_model"] == "gpt2-base.pt"
    assert data["prompt"] == "The capital of France is"
    assert len(data["tokens"]) == 5
    token_texts = [t["text"] for t in data["tokens"]]
    assert token_texts == ["The", "capital", "of", "France", "is"]
    # Token 4 ("is") selected by default
    assert data["selected_token_index"] == 4
    assert data["selected_token"]["text"] == "is"
    assert "Paris" in data["selected_token"]["concept"]
    assert "Paris" in data["default_output"]


def test_story_2_interpreter_reading_internal_meaning(client):
    """User Story 2: The Interpreter (Reading Internal Meaning).

    Clicking any word displays a plain-English concept readout without
    raw layer/neuron IDs or complex activation histograms.
    """
    # Select token 4 ('is')
    res = client.post("/api/interpret", json={"prompt": "The capital of France is", "token_index": 4})
    assert res.status_code == 200
    data = res.json()
    assert data["token"] == "is"
    assert data["concept"] == "Predicting 'Paris' as the capital city of France"
    assert "Paris" in data["tags"] or "Capital Prediction" in data["tags"]
    assert data["activation_score"] >= 0.9

    # Select token 3 ('France')
    res = client.post("/api/interpret", json={"prompt": "The capital of France is", "token_index": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["token"] == "France"
    assert "French Republic" in data["concept"]

    # Invalid token index returns 400
    res = client.post("/api/interpret", json={"prompt": "The capital of France is", "token_index": 999})
    assert res.status_code == 400


def test_story_3_steerer_causal_editing(client):
    """User Story 3: The Steerer (Causal Editing).

    Applies causal steering (e.g. Tokyo / Japan or Suppress Paris) and
    returns Default vs. Steered output side by side.
    """
    # Steer toward Tokyo / Japan
    res = client.post(
        "/api/steer",
        json={
            "prompt": "The capital of France is",
            "concept": "Tokyo / Japan",
            "intensity": 1.0,
            "mode": "boost",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Paris" in data["default_output"]
    assert "Tokyo" in data["steered_output"]
    assert "Tokyo / Japan" in data["difference_summary"]

    # Status reflects the active steer
    status = client.get("/api/status").json()
    assert status["active_steer"] is not None
    assert status["active_steer"]["concept"] == "Tokyo / Japan"
    assert "Tokyo" in status["steered_output"]

    # Suppress Paris
    res = client.post(
        "/api/steer",
        json={
            "prompt": "The capital of France is",
            "concept": "Suppress Paris",
            "intensity": 0.9,
            "mode": "suppress",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Paris" not in data["steered_output"].split("is")[-1]

    # Reset to baseline
    res = client.post("/api/reset")
    assert res.status_code == 200
    status = client.get("/api/status").json()
    assert status["active_steer"] is None
    assert status["steered_output"] == status["default_output"]


def test_story_4_cross_model_save_steer_and_switch(client):
    """User Story 4: The Cross-Model Experimenter (Save Steer & Switch .pt Model).

    Saves a working steer as a preset pill inside Card 2, switches model
    via header switcher / .pt upload, and reuses the preset.
    """
    # 1. Check initial presets
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()
    preset_names = [p["name"] for p in presets]
    assert any("Tokyo Steer" in name for name in preset_names)
    assert any("Berlin Steer" in name for name in preset_names)

    # 2. Save a new preset
    res = client.post(
        "/api/presets",
        json={
            "name": "Rome Steer",
            "concept": "Rome / Italy",
            "intensity": 0.85,
            "mode": "boost",
        },
    )
    assert res.status_code == 200
    new_preset_data = res.json()
    created = new_preset_data["preset"]
    assert "Rome Steer" in created["name"]
    created_id = created["id"]

    # 3. Switch active model via header switcher
    res = client.post("/api/models/select", json={"model": "gpt2-finetuned.pt"})
    assert res.status_code == 200
    status = client.get("/api/status").json()
    assert status["active_model"] == "gpt2-finetuned.pt"

    # Presets are still present for cross-model experimentation
    res = client.get("/api/presets")
    presets = res.json()
    assert any(p["id"] == created_id for p in presets)

    # 4. Upload .pt model checkpoint (placeholder)
    fake_pt_bytes = b"PK\x03\x04torch_checkpoint_placeholder_mock"
    res = client.post(
        "/api/models/upload",
        files={"file": ("custom_finetune.pt", BytesIO(fake_pt_bytes), "application/octet-stream")},
    )
    assert res.status_code == 200
    upload_data = res.json()
    assert upload_data["active_model"] == "custom_finetune.pt"

    # Status reflects uploaded model
    status = client.get("/api/status").json()
    assert status["active_model"] == "custom_finetune.pt"

    # 5. Delete preset
    res = client.delete(f"/api/presets/{created_id}")
    assert res.status_code == 200
    remaining_ids = [p["id"] for p in res.json()["presets"]]
    assert created_id not in remaining_ids


def test_neuronpedia_live_integration(client):
    """Test live Neuronpedia integration when API key is configured."""
    from autoexplain.web.neuronpedia import is_live_configured, fetch_token_activations

    if not is_live_configured():
        pytest.skip("NEURONPEDIA_API_KEY not configured in environment (expected in headless CI).")

    # Test novel prompt interpretation through live API
    res = client.post(
        "/api/interpret",
        json={"prompt": "The Eiffel Tower is in", "token_index": 1},
    )
    assert res.status_code == 200
    data = res.json()
    assert "concept" in data
    assert len(data["concept"]) > 0
    # Status shows live flag
    status = client.get("/api/status").json()
    assert status.get("neuronpedia_live") is True
