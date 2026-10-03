from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Recommendation:
    method: str
    rationale: str
    requirements: str
    candidate_layers: tuple[str, ...] = ()


@contextmanager
def evaluating(model):
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


class Inspector:
    """Inspect an instantiated PyTorch model; never infer an architecture from weights."""

    def __init__(self, model: nn.Module, checkpoint: str | Path | None = None):
        if not isinstance(model, nn.Module):
            raise TypeError("This release supports torch.nn.Module; other adapters are not implemented.")
        self.model = model
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)

    def layers(self) -> dict[str, nn.Module]:
        return dict(self.model.named_modules())

    def suggest(self) -> list[Recommendation]:
        """Structural candidates only; data/task compatibility must still be checked."""
        convs = tuple(name for name, layer in self.layers().items() if isinstance(layer, nn.Conv2d))
        result = [Recommendation(
            "input_gradients", "Differentiable PyTorch model; task compatibility is unverified.",
            "Floating inputs; forward must return [batch, classes] scores; differentiable target.",
        )]
        if convs:
            result.append(Recommendation(
                "gradcam", "Conv2d modules are candidate spatial feature layers, not verified targets.",
                "4D image input, a selected 4D activation used once, and [batch, classes] output.", convs,
            ))
        result.extend([
            Recommendation("integrated_gradients", "Path-integrated input sensitivity with completeness residual.",
                           "Floating inputs, explicit baseline and differentiable class scores; increase steps to check convergence."),
            Recommendation("smoothgrad", "Noise-averaged local input sensitivity.",
                           "Floating inputs, task-appropriate noise scale and fixed targets."),
            Recommendation("occlusion", "Forward-only perturbation attribution.",
                           "Explicit masking baseline and feature/window grouping; inspect distribution shift."),
            Recommendation("activation_patching", "Compare a recipient with matched donor internal features.",
                           "Explicit tensor-output layer, donor alignment, self-patch and random controls."),
        ])
        if convs:
            result.append(Recommendation("cam_backends", "Additional spatial explanation methods via pytorch-grad-cam.",
                           "Install vision extra; select explicit layer and method; EigenCAM is class-independent.", convs))
        result.append(Recommendation(
            "activation_steering", "Tensor-output modules can expose intervention points.",
            "Explicit layer, feature axis, contrastive activations, and matched-control evaluation; "
            "no automatic guarantee of semantic direction or diffusion compatibility.",
        ))
        return result

    @staticmethod
    def _scores(output, target):
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("Expected model output [batch, classes]; wrap structured/segmentation outputs explicitly.")
        if target is None:
            target = output.detach().argmax(dim=1)
        indices = torch.as_tensor(target, device=output.device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.expand(output.shape[0])
        if indices.shape != (output.shape[0],):
            raise ValueError("target must be one class index or one index per sample")
        return output.gather(1, indices[:, None]).sum()

    def input_gradients(self, inputs: torch.Tensor, target=None) -> torch.Tensor:
        """Signed derivative of selected class score w.r.t. input; no parameter grad mutation."""
        with evaluating(self.model), torch.enable_grad():
            x = inputs.detach().clone().requires_grad_(True)
            score = self._scores(self.model(x), target)
            return torch.autograd.grad(score, x)[0].detach()

    def gradcam(self, inputs: torch.Tensor, *, layer: str | None = None, target=None) -> torch.Tensor:
        """Return per-image normalized [B,H,W] Grad-CAM maps for classification scores."""
        if inputs.ndim != 4:
            raise ValueError("Grad-CAM requires [batch, channels, height, width] inputs")
        layers = self.layers()
        if layer is None:
            candidates = [name for name, module in layers.items() if isinstance(module, nn.Conv2d)]
            if not candidates:
                raise ValueError("No Conv2d candidate; specify a spatial tensor-output layer")
            layer = candidates[-1]
        if layer not in layers:
            raise ValueError(f"Unknown layer {layer!r}; use Inspector.layers()")
        activations = []

        def capture(module, args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError("Selected layer must return a 4D tensor")
            # Clone so downstream in-place activations do not overwrite the captured features.
            value = output.clone()
            activations.append(value)
            return value.clone()

        handle = layers[layer].register_forward_hook(capture)
        try:
            with evaluating(self.model), torch.enable_grad():
                x = inputs.detach().clone().requires_grad_(True)
                score = self._scores(self.model(x), target)
                if len(activations) != 1:
                    raise ValueError("Selected layer must execute exactly once per forward")
                activation = activations[0]
                gradients = torch.autograd.grad(score, activation)[0]
                weights = gradients.mean(dim=(-2, -1), keepdim=True)
                cam = (weights * activation).sum(dim=1).relu()
                cam = F.interpolate(cam[:, None], inputs.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                low = cam.amin(dim=(-2, -1), keepdim=True)
                high = cam.amax(dim=(-2, -1), keepdim=True)
                return ((cam - low) / (high - low).clamp_min(1e-12)).detach()
        finally:
            handle.remove()
