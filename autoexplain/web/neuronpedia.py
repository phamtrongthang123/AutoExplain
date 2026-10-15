"""Live Neuronpedia API integration for AutoExplain Studio."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"


def get_api_key() -> Optional[str]:
    """Retrieve Neuronpedia API key from environment or local .env."""
    key = os.getenv("NEURONPEDIA_API_KEY")
    if not key and ENV_PATH.exists():
        try:
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("NEURONPEDIA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        except Exception as e:
            logger.warning("Could not read .env file: %s", e)
    return key or None


def is_live_configured() -> bool:
    """Check if a valid Neuronpedia API key is present."""
    return get_api_key() is not None


def fetch_token_activations(
    text: str,
    model_id: str = "gpt2-small",
    source: str = "7-res-jb",
    num_results: int = 3,
    timeout: float = 6.0,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch live SAE token activations and plain-English explanations from Neuronpedia."""
    api_key = get_api_key()
    if not api_key:
        return None

    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "modelId": model_id,
        "source": source,
        "text": text,
        "numResults": num_results,
    }

    req = urllib.request.Request(
        f"{NEURONPEDIA_BASE_URL}/search-topk-by-token",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return None

            tokens = []
            for item in results:
                pos = item.get("position", len(tokens) + 1)
                raw_tok = item.get("token", "")
                tok_clean = raw_tok.strip() or raw_tok
                top_features = item.get("topFeatures", [])

                if top_features:
                    feat = top_features[0]
                    act_val = float(feat.get("activationValue", 10.0))
                    f_idx = feat.get("featureIndex")
                    exps = feat.get("feature", {}).get("explanations", [])
                    exp_desc = exps[0].get("description") if exps else None
                    concept = exp_desc.strip() if exp_desc else f"Sparse feature #{f_idx} activation"
                    tags = [f"SAE #{f_idx}", f"Layer {source.split('-')[0]}"]
                else:
                    act_val = 5.0
                    f_idx = None
                    concept = f"Contextual representation for '{tok_clean}'"
                    tags = ["General Latent"]

                # Normalize activation score to [0.1, 1.0]
                norm_score = round(min(1.0, max(0.12, act_val / 50.0)), 2)

                tokens.append({
                    "index": pos - 1,
                    "text": tok_clean,
                    "activation_score": norm_score,
                    "concept": concept,
                    "tags": tags,
                    "confidence": 0.95,
                    "feature_index": f_idx,
                })

            return tokens
    except Exception as e:
        logger.warning("Neuronpedia search-topk-by-token error: %s", e)
        return None


def cloud_steer(
    prompt: str,
    feature_index: int = 1049,
    intensity: float = 1.0,
    mode: str = "boost",
    model_id: str = "gpt2-small",
    layer: str = "7-res-jb",
    n_tokens: int = 20,
    timeout: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """Execute live causal steering on Neuronpedia cloud models."""
    api_key = get_api_key()
    if not api_key:
        return None

    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json",
    }

    # Strength calculation: positive for boost, negative for suppress
    strength_val = float(intensity * (25.0 if mode == "boost" else -25.0))

    payload = {
        "prompt": prompt,
        "modelId": model_id,
        "features": [
            {
                "modelId": model_id,
                "layer": layer,
                "index": int(feature_index),
                "strength": strength_val,
            }
        ],
        "n_tokens": n_tokens,
        "temperature": 0.2,
        "strength_multiplier": 1,
        "seed": 42,
        "steer_method": "SIMPLE_ADDITIVE",
    }

    req = urllib.request.Request(
        f"{NEURONPEDIA_BASE_URL}/steer",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            steered_completion = res_data.get("STEERED", "")
            default_completion = res_data.get("DEFAULT", "")
            limit_remaining = res_data.get("limit")

            return {
                "default_output": f"{prompt}{default_completion}",
                "steered_output": f"{prompt}{steered_completion}",
                "remaining_quota": limit_remaining,
                "share_url": res_data.get("shareUrl"),
            }
    except Exception as e:
        logger.warning("Neuronpedia cloud steer error: %s", e)
        return None
