"""Small, explicit primitives for model explanations and interventions."""
from .core import Inspector, Recommendation
from .steering import concept_direction, steer
from .attribution import IntegratedGradientsResult, integrated_gradients, smoothgrad, occlusion, layercam
from .activations import capture_activations, patch_activation
from .metrics import FaithfulnessResult, perturbation_faithfulness
from .nla import (
    NaturalLanguageAutoencoder, NLAActivations, NLAText, NLAReconstruction,
    extract_nla_activations, patch_nla_activations, nla_reconstruction_loss,
)
from .feature_shap import FeatureShapleyResult, feature_shapley, mm_shap, intershap
from .clip_tools import (
    clip_similarity, clip_scores, encode_clip_scores, grad_eclip,
    token_gradient_attribution, patch_attribution_map,
    projected_component_scores, attention_value_decomposition,
)
from .multimodal_concepts import (
    sparse_concept_decomposition, reconstruct_concepts,
    intervene_concepts, modality_feature_diagnostics,
)
from .multimodal_metrics import (
    EvidenceGroup, GroupedEvidenceResult, grouped_evidence_evaluation,
    grounded_evidence_metrics, fixed_prefix_log_probs,
)
from .multimodal_tracing import PatchSite, donor_patching_sweep, cross_modal_attention_payload
from .multimodal import (
    MultimodalScorer, ModalityShapleyResult, ModalityAblationResult,
    MismatchedPairResult, EMAPResult, RelianceComparison,
    modality_shapley, modality_ablations, mismatched_pairs, emap,
    compare_answer_explanation_reliance,
)

__all__ = [
    "Inspector", "Recommendation", "concept_direction", "steer",
    "IntegratedGradientsResult", "integrated_gradients", "smoothgrad", "occlusion", "layercam",
    "capture_activations", "patch_activation", "FaithfulnessResult", "perturbation_faithfulness",
    "MultimodalScorer", "ModalityShapleyResult", "ModalityAblationResult",
    "MismatchedPairResult", "EMAPResult", "RelianceComparison",
    "modality_shapley", "modality_ablations", "mismatched_pairs", "emap",
    "compare_answer_explanation_reliance",
    "FeatureShapleyResult", "feature_shapley", "mm_shap", "intershap",
    "clip_similarity", "clip_scores", "encode_clip_scores", "grad_eclip",
    "token_gradient_attribution", "patch_attribution_map",
    "projected_component_scores", "attention_value_decomposition",
    "sparse_concept_decomposition", "reconstruct_concepts",
    "intervene_concepts", "modality_feature_diagnostics",
    "EvidenceGroup", "GroupedEvidenceResult", "grouped_evidence_evaluation",
    "grounded_evidence_metrics", "fixed_prefix_log_probs",
    "PatchSite", "donor_patching_sweep", "cross_modal_attention_payload",
    "NaturalLanguageAutoencoder", "NLAActivations", "NLAText", "NLAReconstruction",
    "extract_nla_activations", "patch_nla_activations", "nla_reconstruction_loss",
]
