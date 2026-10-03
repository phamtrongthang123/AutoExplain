"""Small, explicit primitives for model explanations and interventions."""
from .core import Inspector, Recommendation
from .steering import concept_direction, steer
from .attribution import IntegratedGradientsResult, integrated_gradients, smoothgrad, occlusion, layercam
from .activations import capture_activations, patch_activation
from .metrics import FaithfulnessResult, perturbation_faithfulness

__all__ = [
    "Inspector", "Recommendation", "concept_direction", "steer",
    "IntegratedGradientsResult", "integrated_gradients", "smoothgrad", "occlusion", "layercam",
    "capture_activations", "patch_activation", "FaithfulnessResult", "perturbation_faithfulness",
]
