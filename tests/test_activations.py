import pytest
import torch
from torch import nn

from autoexplain.activations import capture_activations, patch_activation


def test_capture_snapshots_repeated_invocations_and_inplace():
    model = nn.Sequential(nn.Identity(), nn.ReLU(inplace=True))
    model.train()
    with capture_activations(model, ["0"]) as values:
        model(torch.tensor([[-1., 2.]]))
        model(torch.tensor([[-3., 4.]]))
    assert len(values["0"]) == 2
    assert torch.equal(values["0"][0], torch.tensor([[-1., 2.]]))
    assert model.training and not model[0]._forward_hooks
    assert not values["0"][0].requires_grad


def test_patch_mask_formula_and_restoration():
    model = nn.Sequential(nn.Identity())
    x = torch.tensor([[1., 2.], [3., 4.]])
    with patch_activation(model, "0", torch.zeros_like(x), mask=torch.tensor([True, False])):
        assert torch.equal(model(x), torch.tensor([[0., 2.], [0., 4.]]))
    assert torch.equal(model(x), x)
    assert not model[0]._forward_hooks


def test_nested_capture_patch_and_exception_cleanup():
    model = nn.Sequential(nn.Identity())
    with pytest.raises(RuntimeError):
        with capture_activations(model, "0") as values:
            with patch_activation(model, "0", torch.zeros(1, 2)):
                model(torch.ones(1, 2))
                raise RuntimeError("failure")
    assert torch.equal(values["0"][0], torch.ones(1, 2))
    assert not model[0]._forward_hooks


def test_validation_cleanup():
    model = nn.Sequential(nn.Identity())
    with pytest.raises(ValueError, match="shape"):
        with patch_activation(model, "0", torch.zeros(2, 3)):
            model(torch.ones(1, 2))
    assert not model[0]._forward_hooks
    with pytest.raises(ValueError, match="Unknown"):
        with capture_activations(model, ["0", "missing"]):
            pass
    assert not model[0]._forward_hooks


def test_context_does_not_mutate_existing_parameter_grads_or_modes():
    model = nn.Sequential(nn.Linear(2, 2))
    model.train()
    model[0].eval()
    model[0].weight.grad = torch.ones_like(model[0].weight)
    with capture_activations(model, "0"):
        with patch_activation(model, "0", torch.zeros(1, 2)):
            model(torch.ones(1, 2))
    assert model.training and not model[0].training
    assert torch.equal(model[0].weight.grad, torch.ones_like(model[0].weight))
