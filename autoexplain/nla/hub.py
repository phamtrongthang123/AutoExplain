"""Hugging Face Hub utilities for Natural Language Autoencoder (NLA) checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from huggingface_hub import hf_hub_download, snapshot_download

from autoexplain.nla.config import NLAConfig, load_nla_config

logger = logging.getLogger(__name__)

# Canonical checkpoints published under kitft/
KNOWN_NLA_MODELS: Dict[str, Dict[str, str]] = {
    "qwen2.5-7b-L20": {
        "av": "kitft/nla-qwen2.5-7b-L20-av",
        "ar": "kitft/nla-qwen2.5-7b-L20-ar",
        "layer": "20",
        "d_model": "3584",
    },
    "gemma3-12b-L32": {
        "av": "kitft/nla-gemma3-12b-L32-av",
        "ar": "kitft/nla-gemma3-12b-L32-ar",
        "layer": "32",
        "d_model": "3840",
    },
    "gemma3-27b-L41": {
        "av": "kitft/nla-gemma3-27b-L41-av",
        "ar": "kitft/nla-gemma3-27b-L41-ar",
        "layer": "41",
        "d_model": "4608",
    },
    "llama-3.3-70b-L53": {
        "av": "kitft/Llama-3.3-70B-NLA-L53-av",
        "ar": "kitft/Llama-3.3-70B-NLA-L53-ar",
        "layer": "53",
        "d_model": "8192",
    },
}


def list_available_nla_checkpoints() -> List[str]:
    """List known pretrained NLA model keys."""
    return list(KNOWN_NLA_MODELS.keys())


def load_nla_metadata_from_hub(repo_id: str) -> NLAConfig:
    """Download nla_meta.yaml from Hugging Face Hub and parse into NLAConfig."""
    yaml_path = hf_hub_download(repo_id=repo_id, filename="nla_meta.yaml")
    return load_nla_config(yaml_path)


def download_nla_checkpoint(repo_id: str, local_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download full NLA checkpoint from Hugging Face Hub."""
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir) if local_dir else None,
        allow_patterns=["*.json", "*.yaml", "*.safetensors", "*.jinja", "*.txt"],
    )
    return Path(downloaded_path)
