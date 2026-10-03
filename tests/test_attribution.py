import pytest
import torch
from torch import nn

from autoexplain.attribution import integrated_gradients, layercam, occlusion, smoothgrad


def linear():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2., 3.], [-2., 0., 1.]]))
        model.bias.fill_(7)
    return model


def test_ig_linear_completeness_and_per_sample_targets():
    model = linear()
    x = torch.tensor([[2., 3., 4.], [4., 2., 3.]])
    result = integrated_gradients(model, x, torch.tensor([0, 1]), baseline=1., steps=4)
    assert torch.allclose(result.attributions, (x - 1) * model.weight)
    assert torch.allclose(result.completeness_delta, torch.zeros(2), atol=1e-6)


def test_ig_quadratic_trapezoidal_exact():
    class Square(nn.Module):
        def forward(self, x):
            return x.square().sum(1, keepdim=True)
    x = torch.tensor([[1., 2., 3.]])
    result = integrated_gradients(Square(), x, steps=8)
    assert torch.allclose(result.attributions, x.square())
    assert result.completeness_delta.abs().max() < 1e-6


def test_smoothgrad_linear_and_seed():
    model = linear()
    x = torch.ones(2, 3)
    result = smoothgrad(model, x, 1, samples=8)
    assert torch.allclose(result, model.weight[1].expand_as(x))
    nonlinear = nn.Sequential(nn.Linear(3, 2), nn.Sigmoid())
    a = smoothgrad(nonlinear, x, 0, generator=torch.Generator().manual_seed(42))
    b = smoothgrad(nonlinear, x, 0, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)


def test_occlusion_single_and_overlap_formula():
    model = linear()
    x = torch.tensor([[1., 2., 3.]])
    assert torch.allclose(occlusion(model, x, 0), x * model.weight[0])
    assert torch.allclose(occlusion(model, x, 0, window_shape=(2,)), torch.tensor([[5., 9., 13.]]))


def test_layercam_formula_inplace_frozen_and_cleanup():
    model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.ReLU(inplace=True),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
    with torch.no_grad():
        model[0].weight.fill_(1)
    model.requires_grad_(False)
    x = torch.arange(4.).reshape(1, 1, 2, 2)
    assert torch.allclose(layercam(model, x, layer="0", normalize=False), x[:, 0] / 4)
    assert torch.allclose(layercam(model, x, layer="0"), x[:, 0] / 3)
    assert not model[0]._forward_hooks


@pytest.mark.parametrize("method", [integrated_gradients, smoothgrad, occlusion])
def test_state_grads_inputs_preserved(method):
    model = nn.Sequential(linear(), nn.Identity())
    model.train()
    model[0].eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    x = torch.ones(2, 3)
    method(model, x, 0)
    assert model.training and not model[0].training
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert not x.requires_grad and torch.equal(x, torch.ones_like(x))


def test_errors_restore_modes_and_hooks():
    model = nn.Sequential(nn.Conv2d(1, 1, 1))
    model.train()
    with pytest.raises(ValueError, match="output"):
        layercam(model, torch.ones(1, 1, 2, 2), layer="0")
    assert model.training and not model[0]._forward_hooks
    with pytest.raises(ValueError, match="positive"):
        integrated_gradients(linear(), torch.ones(1, 3), steps=0)
    with pytest.raises(ValueError, match="stride"):
        occlusion(linear(), torch.ones(1, 3), window_shape=(1,), strides=(2,))


def test_target_none_is_fixed_to_clean_input():
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.], [-1.]]))
        model.bias.zero_()
    result = integrated_gradients(model, torch.tensor([[2.]]), baseline=-2., steps=8)
    assert torch.equal(result.attributions, torch.tensor([[4.]]))
    assert torch.equal(result.completeness_delta, torch.zeros(1))
