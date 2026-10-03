"""Lazy integrations with Captum, pytorch-grad-cam and SHAP.

Upstream packages retain their licenses. Nothing is downloaded by these wrappers.
"""
import importlib
import numpy as np
import torch
from .core import evaluating

CAPTUM_METHODS = {
    "saliency": "Saliency", "input_x_gradient": "InputXGradient",
    "integrated_gradients": "IntegratedGradients", "deeplift": "DeepLift",
    "gradient_shap": "GradientShap", "deeplift_shap": "DeepLiftShap",
    "guided_backprop": "GuidedBackprop", "deconvolution": "Deconvolution",
    "feature_ablation": "FeatureAblation", "feature_permutation": "FeaturePermutation",
    "occlusion": "Occlusion", "lime": "Lime", "kernel_shap": "KernelShap",
    "shapley_sampling": "ShapleyValueSampling", "shapley_exact": "ShapleyValues",
}
CAM_METHODS = {
    "gradcam": "GradCAM", "gradcam_plus_plus": "GradCAMPlusPlus",
    "hirescam": "HiResCAM", "scorecam": "ScoreCAM", "ablationcam": "AblationCAM",
    "eigencam": "EigenCAM", "eigengradcam": "EigenGradCAM", "layercam": "LayerCAM",
    "fullgrad": "FullGrad", "xgradcam": "XGradCAM", "elementwise_gradcam": "GradCAMElementWise",
}
LAYER_METHODS = {"layer_integrated_gradients": "LayerIntegratedGradients",
                 "layer_gradient_x_activation": "LayerGradientXActivation",
                 "layer_conductance": "LayerConductance", "layer_ablation": "LayerFeatureAblation"}


def require(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"Install autoexplain-ai[{extra}] and its compatible dependencies for {module}") from exc


def captum_attribute(model, inputs, *, method, target=None, baseline=None,
                     feature_mask=None, **kwargs):
    """Call a Captum method with explicit kwargs; output follows Captum's contract.

    Baseline distributions are required for GradientSHAP/DeepLiftSHAP. Feature
    permutation needs multiple independent samples. Group features before exact
    Shapley/LIME/KernelSHAP. Expensive methods are refused above small budgets.
    """
    if method not in CAPTUM_METHODS:
        raise ValueError(f"Unknown method; choose from {sorted(CAPTUM_METHODS)}")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2 or not inputs.numel():
        raise ValueError("Expected a nonempty batched input tensor")
    if method == "feature_permutation" and inputs.shape[0] < 2:
        raise ValueError("Feature permutation requires at least two samples")
    if feature_mask is not None:
        if feature_mask.dtype not in (torch.int32, torch.int64) or (feature_mask < 0).any():
            raise ValueError("feature_mask needs nonnegative integer group IDs")
        torch.broadcast_shapes(feature_mask.shape, inputs.shape)
        groups = int(feature_mask.max()) + 1
    else:
        groups = inputs[0].numel()
    if method == "shapley_exact" and groups > 6:
        raise ValueError("Exact Shapley limited to six feature groups; use sampling")
    if method in {"kernel_shap", "lime", "shapley_sampling"} and groups > 64:
        raise ValueError("Group inputs into at most 64 features for this local wrapper")
    if method in {"gradient_shap", "deeplift_shap"} and baseline is None:
        raise ValueError("This method needs an explicit reference distribution")
    attr = require("captum.attr", "attribution")
    arguments = dict(kwargs, target=target)
    if baseline is not None:
        arguments["baselines"] = baseline
    if feature_mask is not None:
        arguments["feature_mask"] = feature_mask
    with evaluating(model), torch.enable_grad():
        result = getattr(attr, CAPTUM_METHODS[method])(model).attribute(inputs.detach().clone(), **arguments)
    if isinstance(result, tuple):
        return tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in result)
    return result.detach()


def captum_layer(model, inputs, *, layer, method, target=None, **kwargs):
    if method not in LAYER_METHODS:
        raise ValueError("Unknown layer method")
    module = dict(model.named_modules()).get(layer)
    if module is None:
        raise ValueError("Unknown layer")
    attr = require("captum.attr", "attribution")
    with evaluating(model), torch.enable_grad():
        result = getattr(attr, LAYER_METHODS[method])(model, module).attribute(inputs, target=target, **kwargs)
    return result.detach() if isinstance(result, torch.Tensor) else result


def cam_attribute(model, inputs, *, layers, method="gradcam", target=None,
                  reshape_transform=None, eigen_smooth=False, aug_smooth=False):
    """Return CPU [B,H,W] maps. Target is a scalar class or one index per sample.

    EigenCAM is class-independent. FullGrad uses upstream bias-layer discovery.
    reshaping transformer activations is an explicit caller responsibility.
    """
    if method not in CAM_METHODS:
        raise ValueError("Unknown CAM method")
    if inputs.ndim != 4 or not inputs.is_floating_point():
        raise ValueError("CAM input must be floating [B,C,H,W]")
    modules = dict(model.named_modules())
    if not layers or any(name not in modules for name in layers):
        raise ValueError("Supply valid target layer names")
    backend = require("pytorch_grad_cam", "vision")
    targets = None
    if target is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        indices = [target] * inputs.shape[0] if isinstance(target, int) else list(target)
        if len(indices) != inputs.shape[0] or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Target must be nonnegative class indices matching batch")
        targets = [ClassifierOutputTarget(i) for i in indices]
    # Upstream calls backward(); preserve pre-existing gradients, including None.
    grads = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    try:
        with evaluating(model), torch.enable_grad():
            with getattr(backend, CAM_METHODS[method])(
                    model=model, target_layers=[modules[n] for n in layers],
                    reshape_transform=reshape_transform) as explainer:
                result = explainer(input_tensor=inputs.detach().clone(), targets=targets,
                                   eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        return torch.from_numpy(np.asarray(result).copy())
    finally:
        for param, grad in grads:
            param.grad = grad


def attribution_infidelity(model, inputs, attributions, perturb_func, *, target=None, samples=16):
    """Captum infidelity with caller-defined perturbation and explicit target."""
    metrics = require("captum.metrics", "attribution")
    if not 1 <= samples <= 1024:
        raise ValueError("samples must be 1–1024")
    with evaluating(model):
        return metrics.infidelity(model, perturb_func, inputs, attributions,
                                  target=target, n_perturb_samples=samples).detach()


def attribution_sensitivity(explain_func, inputs, *, samples=8, radius=0.02, **kwargs):
    """Captum max sensitivity over local input perturbations; not a faithfulness proof."""
    metrics = require("captum.metrics", "attribution")
    if not 1 <= samples <= 1024 or not 0 < radius <= 1:
        raise ValueError("Invalid sensitivity budget")
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("This wrapper accepts one input tensor")
    def bridge(values, **arguments):
        # Captum normalizes even a single tensor to a one-element tuple.
        x = values[0] if isinstance(values, tuple) else values
        result = explain_func(x, **arguments)
        return result if isinstance(result, tuple) else (result,)
    return metrics.sensitivity_max(bridge, inputs, n_perturb_samples=samples,
                                   perturb_radius=radius, **kwargs).detach()


def shap_explain(model_or_predict, inputs, *, method="tree", background=None, **kwargs):
    """Return native SHAP Explanation; kwargs go to the explainer's call.

    tree needs a supported fitted tree; permutation/exact need a callable and
    background. Feature count is capped for the exponential exact explainer.
    """
    shap = require("shap", "tabular")
    x = np.asarray(inputs)
    if x.ndim != 2 or not x.size:
        raise ValueError("SHAP input must be a nonempty 2D array")
    if method == "tree":
        return shap.TreeExplainer(model_or_predict, data=background)(x, **kwargs)
    if method not in {"permutation", "exact", "linear"}:
        raise ValueError("method must be tree, permutation, exact or linear")
    if background is None:
        raise ValueError("Explicit background required")
    if method == "exact" and x.shape[1] > 10:
        raise ValueError("Exact SHAP limited to ten features")
    cls = {"permutation": shap.PermutationExplainer, "exact": shap.ExactExplainer,
           "linear": shap.LinearExplainer}[method]
    return cls(model_or_predict, background)(x, **kwargs)
