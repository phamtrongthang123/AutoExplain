"""Unit tests for Natural Language Autoencoders (NLA) module."""

import math
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from autoexplain.core.adapter import ModelAdapter
from autoexplain.nla.config import (
    NLAConfig,
    inject_at_marked_positions,
    load_nla_config,
    normalize_activation,
)
from autoexplain.nla.hub import KNOWN_NLA_MODELS, list_available_nla_checkpoints
from autoexplain.nla.reconstructor import ActivationReconstructor, NLACritic
from autoexplain.nla.tool import NLATool, NaturalLanguageAutoencoder
from autoexplain.nla.verbalizer import ActivationVerbalizer
from autoexplain.tools.registry import ToolRegistry


class DummyLM(nn.Module):
    """Simple 2-layer module with residual connection simulating a transformer block."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(50, d_model)
        self.block0 = nn.Linear(d_model, d_model)
        self.block1 = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 50)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = h + self.block0(h)
        h = h + self.block1(h)
        return self.head(h)


def test_nla_normalization():
    v = torch.randn(1, 128)
    v_norm = normalize_activation(v, target_scale=150.0)
    assert pytest.approx(v_norm.norm(dim=-1).item(), rel=1e-3) == 150.0


def test_nla_marked_injection():
    input_ids = torch.tensor([[10, 29, 999, 522, 20]])  # Token 999 surrounded by 29 and 522
    embeds = torch.zeros(1, 5, 8)
    vector = torch.ones(1, 8) * 3.14

    injected = inject_at_marked_positions(
        input_ids=input_ids,
        embeddings=embeds,
        vectors=vector,
        inj_id=999,
        left_id=29,
        right_id=522,
    )

    # Position 2 should be overwritten
    assert torch.allclose(injected[0, 2], vector[0])
    # Position 0 should still be zero
    assert torch.allclose(injected[0, 0], torch.zeros(8))


def test_nla_config_loading():
    meta_dict = {
        "kind": "nla_model",
        "role": "av",
        "d_model": 128,
        "extraction": {
            "injection_scale": 150.0,
            "mse_scale": 11.31,
        },
        "tokens": {
            "injection_char": "㈎",
            "injection_token_id": 100,
            "injection_left_neighbor_id": 20,
            "injection_right_neighbor_id": 30,
        },
        "prompt_templates": {
            "av": "Explain: <concept>{injection_char}</concept>",
            "ar": "Summary: <text>{explanation}</text>",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = Path(tmpdir) / "nla_meta.yaml"
        meta_file.write_text(yaml.dump(meta_dict))

        cfg = load_nla_config(tmpdir)
        assert cfg.d_model == 128
        assert cfg.injection_scale == 150.0
        assert cfg.injection_char == "㈎"
        assert cfg.injection_token_id == 100


def test_verbalizer_tag_extraction():
    verb = ActivationVerbalizer()
    raw = "Here is the result: <explanation>Subject entity performing mathematical logic</explanation> trailing text."
    extracted = verb.extract_explanation_tags(raw)
    assert extracted == "Subject entity performing mathematical logic"


def test_reconstructor_scoring_and_control():
    d = 32
    critic = ActivationReconstructor(
        config=NLAConfig(
            d_model=d,
            injection_char="㈎",
            injection_token_id=1,
            injection_left_neighbor_id=2,
            injection_right_neighbor_id=3,
            actor_prompt_template="",
            critic_prompt_template="",
            injection_scale=10.0,
            mse_scale=math.sqrt(d),
        )
    )

    gold = torch.randn(d)
    scores = critic.score("Test explanation", gold)
    assert "mse" in scores
    assert "cosine_similarity" in scores
    # Mathematical relation MSE = 2 * (1 - cos)
    expected_mse = 2.0 * (1.0 - scores["cosine_similarity"])
    assert pytest.approx(scores["mse"], rel=1e-2) == expected_mse

    # Control verification
    control = critic.verify_against_control("Test explanation", gold, num_random_trials=3)
    assert "verified" in control
    assert "random_control_mean_cosine" in control


def test_nla_tool_end_to_end_investigation():
    model = DummyLM(d_model=64)
    adapter = ModelAdapter(model=model, device="cpu")

    tool = NLATool()
    assert tool.name == "nla"
    assert "causal_lm" in tool.supported_model_types

    dummy_inputs = torch.randint(0, 40, (1, 6))
    res = tool.run(adapter, inputs=dummy_inputs, target_layer="block1")

    assert res.tool_name == "nla"
    assert "cosine_similarity" in res.metrics
    assert "mse" in res.metrics
    assert len(res.findings) >= 2
    assert "NLA Verbalization" in res.findings[0]


def test_tool_registry_has_nla():
    assert "nla" in ToolRegistry.list_tools()
    tool_cls = ToolRegistry.get("nla")
    assert tool_cls == NLATool


def test_hub_known_models():
    models = list_available_nla_checkpoints()
    assert "qwen2.5-7b-L20" in models
    assert "gemma3-12b-L32" in models
