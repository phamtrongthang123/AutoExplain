"""Tests proving acceptance criteria for AutoExplain Harness on GPT-2."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from autoexplain import Harness


@pytest.fixture(scope="module")
def harness():
    """Create a persistent Harness instance for test execution."""
    h = Harness(device="cpu")
    h.open_model("gpt2")
    return h


@pytest.fixture(autouse=True)
def clean_state(harness):
    """Ensure harness interventions are cleared before and after each test."""
    harness.reset()
    yield
    harness.reset()


def test_gpt2_loads_successfully(harness):
    """Proves GPT-2 loads successfully into memory."""
    assert harness.model is not None
    assert harness.model_name == "gpt2"


def test_model_instance_persists_between_runs(harness):
    """Proves the exact model instance persists across repeated runs without reloading."""
    model_obj_id = id(harness.model)

    harness.run("The capital of France is")
    assert id(harness.model) == model_obj_id

    # Calling open_model with the same name reuses the instance
    harness.open_model("gpt2")
    assert id(harness.model) == model_obj_id


def test_inspect_exposes_layers_and_heads(harness):
    """Proves inspect() exposes architecture, layer count, and attention heads."""
    info = harness.inspect()
    assert info["model_name"] == "gpt2"
    assert info["n_layers"] == 12
    assert info["n_heads"] == 12
    assert info["d_model"] == 768
    assert info["d_head"] == 64
    assert len(info["layers"]) == 12
    assert info["heads_per_layer"] == 12


def test_observe_retrieves_real_internal_information(harness):
    """Proves observe() retrieves real internal norms and head summaries."""
    prompt = "The sky is"
    harness.run(prompt)
    obs = harness.observe(layer=4)

    assert obs["layer"] == 4
    assert obs["prompt"] == prompt
    assert obs["residual_norm"] > 0.0
    assert len(obs["head_norms"]) == 12
    assert all(norm > 0.0 for norm in obs["head_norms"].values())
    assert obs["attention_output_norm"] > 0.0
    assert len(obs["top_predictions"]) > 0


def test_attention_head_ablation_executes_correctly(harness):
    """Proves genuine attention-head ablation alters internal computations and logits."""
    prompt = "Alice gave Bob a book."
    baseline = harness.run(prompt)

    harness.intervene(layer=5, head=3, method="zero")
    modified = harness.run(prompt)

    diff = (baseline.logits - modified.logits).abs().max().item()
    assert diff > 0.05, f"Expected ablation to alter logits, but max diff was {diff}"


def test_evaluate_measures_actual_output_differences(harness):
    """Proves evaluate() computes actual numerical differences between runs."""
    prompt = "Alice gave Bob a book."
    baseline = harness.run(prompt)

    harness.intervene(layer=5, head=3, method="zero")
    modified = harness.run(prompt)

    res = harness.evaluate(baseline, modified)
    assert res["max_logit_diff"] > 0.05
    assert res["mean_logit_diff"] > 0.0
    assert res["cosine_similarity"] < 1.0
    assert res["cosine_similarity"] > 0.9  # Should still be highly correlated
    assert res["kl_divergence"] > 0.0


def test_reset_restores_baseline_within_numerical_tolerance(harness):
    """Proves reset() restores exact baseline behavior without reloading weights."""
    prompt = "Alice gave Bob a book."
    baseline = harness.run(prompt)

    harness.intervene(layer=5, head=3, method="zero")
    harness.run(prompt)

    harness.reset()
    restored = harness.run(prompt)

    # Logits should match baseline within float numerical precision
    assert torch.allclose(baseline.logits, restored.logits, atol=1e-5), (
        f"Reset failed to restore baseline: max error {(baseline.logits - restored.logits).abs().max().item()}"
    )


def test_record_saves_valid_experiment(harness):
    """Proves record() writes a valid JSON experiment record to disk."""
    prompt = "Alice gave Bob a book."
    baseline = harness.run(prompt)

    harness.intervene(layer=5, head=3, method="zero")
    modified = harness.run(prompt)
    harness.evaluate(baseline, modified)
    harness.reset()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_exp.json"
        saved = harness.record("test_exp", path=out_file)
        assert saved.exists()

        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "test_exp"
        assert data["model"] == "gpt2"
        assert data["prompt"] == prompt
        assert len(data["interventions"]) == 1
        assert data["interventions"][0]["layer"] == 5
        assert data["interventions"][0]["head"] == 3
        assert data["evaluation"]["max_logit_diff"] > 0.0
        assert len(data["baseline_top_tokens"]) > 0
        assert len(data["modified_top_tokens"]) > 0


def test_p0_cache_invalidation_on_reset_and_intervene(harness):
    """Proves reset() and intervene() invalidate activation cache so observe() is never stale."""
    prompt = "Alice gave Bob a book."

    # Baseline observation: head 3 has nonzero norm
    obs_base = harness.observe(layer=5, prompt=prompt)
    assert obs_base["head_norms"][3] > 0.0

    # Intervene: zero head 3
    harness.intervene(layer=5, head=3, method="zero")
    assert harness._last_cache is None, "intervene() must invalidate cache"

    # Observe under intervention: head 3 norm must be 0.0
    obs_ablated = harness.observe(layer=5, prompt=prompt)
    assert pytest.approx(obs_ablated["head_norms"][3], abs=1e-5) == 0.0

    # Reset: must invalidate cache so next observe recaptures clean baseline
    harness.reset()
    assert harness._last_cache is None, "reset() must invalidate cache"

    obs_restored = harness.observe(layer=5, prompt=prompt)
    assert obs_restored["head_norms"][3] > 0.0, "observe() after reset must not return stale ablated cache"
    assert pytest.approx(obs_restored["head_norms"][3], rel=1e-4) == obs_base["head_norms"][3]


def test_p1_interventions_tied_to_specific_evaluated_run(harness):
    """Proves record() uses interventions tied to the evaluated run, preventing cross-experiment leakage."""
    prompt = "Alice gave Bob a book."
    baseline = harness.run(prompt)

    # Experiment 1: ablate layer 5 head 3
    harness.intervene(layer=5, head=3, method="zero")
    modified_1 = harness.run(prompt)
    assert len(modified_1.interventions) == 1
    assert modified_1.interventions[0]["layer"] == 5

    harness.evaluate(baseline, modified_1)
    harness.reset()

    # Experiment 2 is configured subsequently
    harness.intervene(layer=8, head=1, method="zero")
    modified_2 = harness.run(prompt)
    assert modified_2.interventions[0]["layer"] == 8

    # Recording experiment 1 must record layer 5 head 3, NOT experiment 2's layer 8 head 1
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "exp1.json"
        harness.record("exp1", path=out_file)

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(data["interventions"]) == 1
        assert data["interventions"][0]["layer"] == 5, (
            f"Expected layer 5 from evaluated run, but got {data['interventions'][0]['layer']}"
        )


def test_p2_run_does_not_allocate_all_layers_cache(harness):
    """Proves run() does not populate full-model activation cache, and observe() targets only the requested layer."""
    prompt = "Alice gave Bob a book."
    harness.reset()

    # Standard run should not store entire-model activation cache
    harness.run(prompt)
    assert harness._last_cache is None, "run() must not cache all layer activations"

    # observe() on layer 5 should only capture layer 5 activations
    obs = harness.observe(layer=5, prompt=prompt)
    assert harness._last_cache is not None
    assert all("blocks.5." in k for k in harness._last_cache.keys()), "observe() must only capture requested layer"

