"""Small representation diagnostics and controlled interventions, not semantic proof."""
from contextlib import contextmanager
import torch
from .core import evaluating


def _matrix(x):
    if x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point() or not x.isfinite().all():
        raise ValueError("Expected finite nonempty floating [observations, features]")
    return x.detach().double()


def linear_cka(x, y):
    """Biased centered linear CKA; same observations, possibly different widths."""
    x, y = _matrix(x), _matrix(y)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("Need at least two matched observations")
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = (x.T @ x).norm() * (y.T @ y).norm()
    if denominator <= 1e-15:
        raise ValueError("CKA undefined for a constant representation")
    return ((x.T @ y).square().sum() / denominator).item()


def ridge_probe(train_x, train_y, test_x, *, alpha=1.0):
    """Train-only standardization and ridge classification; returns test labels.

    Choose alpha using a separate validation split, not test outcomes.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    probe.fit(_matrix(train_x).cpu().numpy(), train_y.detach().cpu().numpy())
    return probe, probe.predict(_matrix(test_x).cpu().numpy())


def concept_sensitivity(gradients, direction):
    """Fraction of positive directional derivatives (TCAV score primitive).

    Requires independent CAV training, random-concept controls and significance
    tests for a TCAV study; this statistic alone is not the complete method.
    """
    gradients = _matrix(gradients)
    direction = torch.as_tensor(direction, device=gradients.device, dtype=gradients.dtype)
    if direction.shape != (gradients.shape[1],) or not direction.isfinite().all() or direction.norm() == 0:
        raise ValueError("Direction must be finite nonzero and match feature width")
    return ((gradients @ direction) > 0).double().mean().item()


def attention_rollout(attentions, *, add_residual=True):
    """Mean-head attention rollout [B,T,T]; descriptive, not causal attribution."""
    if not attentions:
        raise ValueError("Need attention matrices")
    result = None
    shape = None
    for tensor in attentions:
        a = tensor.detach()
        if a.ndim != 4 or a.shape[-1] != a.shape[-2] or not a.isfinite().all() or (a < 0).any():
            raise ValueError("Expected nonnegative finite [B,heads,T,T]")
        a = a.mean(1)
        if shape is not None and a.shape != shape:
            raise ValueError("Attention sequence shapes must agree")
        shape = a.shape
        if add_residual:
            a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        if (a.sum(-1) <= 0).any():
            raise ValueError("Attention rows must have positive mass")
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    return result


def logit_lens(hidden_states, unembed, *, final_norm=None):
    """Read hidden states through caller-supplied final norm and unembedding."""
    with torch.no_grad():
        return [unembed(final_norm(h) if final_norm is not None else h).detach() for h in hidden_states]


@contextmanager
def ablate_features(model, layer, indices, *, axis=-1, value=0.0):
    """Replace selected neurons/heads on an explicit tensor axis; hooks are scoped."""
    modules = dict(model.named_modules())
    if layer not in modules or not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Need valid layer and nonnegative indices")
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or not -output.ndim <= axis < output.ndim:
            raise ValueError("Need tensor output and valid feature axis")
        if max(indices) >= output.shape[axis]:
            raise ValueError("Feature index outside output")
        changed = output.clone()
        select = [slice(None)] * output.ndim
        select[axis] = list(indices)
        changed[tuple(select)] = value
        return changed
    handle = modules[layer].register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def parameter_randomization_check(model_factory, original, inputs, explain, *, seed=7):
    """Compare flattened explanations with an independently initialized same architecture.

    model_factory must construct a fresh model. Return cosine similarity, not a
    pass/fail scientific verdict; original parameters are never randomized in place.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_model = model_factory()
    if random_model is original:
        raise ValueError("Factory must return a distinct model")
    with evaluating(original), evaluating(random_model):
        a, b = explain(original, inputs).detach().flatten(1), explain(random_model, inputs).detach().flatten(1)
    if a.shape != b.shape or (a.norm(dim=1) == 0).any() or (b.norm(dim=1) == 0).any():
        raise ValueError("Need equal-sized nonzero explanations")
    return torch.nn.functional.cosine_similarity(a, b, dim=1)
