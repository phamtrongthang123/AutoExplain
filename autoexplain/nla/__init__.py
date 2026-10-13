"""Natural Language Autoencoders (NLA) interpretability module."""

from autoexplain.nla.config import (
    NLAConfig,
    inject_at_marked_positions,
    load_nla_config,
    normalize_activation,
)
from autoexplain.nla.hub import (
    KNOWN_NLA_MODELS,
    list_available_nla_checkpoints,
    load_nla_metadata_from_hub,
)
from autoexplain.nla.reconstructor import ActivationReconstructor, NLACritic
from autoexplain.nla.tool import NLATool, NaturalLanguageAutoencoder
from autoexplain.nla.verbalizer import ActivationVerbalizer

__all__ = [
    "NLAConfig",
    "load_nla_config",
    "normalize_activation",
    "inject_at_marked_positions",
    "ActivationVerbalizer",
    "ActivationReconstructor",
    "NLACritic",
    "NaturalLanguageAutoencoder",
    "NLATool",
    "KNOWN_NLA_MODELS",
    "list_available_nla_checkpoints",
    "load_nla_metadata_from_hub",
]
