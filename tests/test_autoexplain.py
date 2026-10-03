import pytest
import torch
from torch import nn

from autoexplain import Inspector, concept_direction, steer
from autoexplain.demo import BarNet


def test_input_gradient_exact_and_state_preserved():
    model = nn.Linear(3, 2)
    model.train()
    model.weight.grad = torch.ones_like(model.weight)
    expected_grad = model.weight.grad.clone()
    x = torch.randn(4, 3)
    gradient = Inspector(model).input_gradients(x, target=1)
    assert torch.allclose(gradient, model.weight[1].expand_as(x))
    assert torch.equal(model.weight.grad, expected_grad)
    assert model.training
    assert not x.requires_grad


def test_cam_known_formula_and_cleanup():
    model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model[0].weight.data.fill_(1)
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    cam = Inspector(model).gradcam(x)
    assert torch.allclose(cam, x[:, 0] / 15)
    assert not model[0]._forward_hooks


def test_cam_inplace_relu_frozen_parameters_and_mixed_mode():
    model = BarNet()
    model.features[1] = nn.ReLU(inplace=True)
    model.features[3] = nn.ReLU(inplace=True)
    model.train()
    model.features[0].eval()
    model.requires_grad_(False)
    cam = Inspector(model).gradcam(torch.randn(2, 1, 16, 16))
    assert cam.shape == (2, 16, 16)
    assert cam.isfinite().all() and cam.min() >= 0 and cam.max() <= 1
    assert model.training and not model.features[0].training


def test_failure_removes_cam_hook():
    model = nn.Sequential(nn.Conv2d(1, 2, 1))
    with pytest.raises(ValueError, match="output"):
        Inspector(model).gradcam(torch.randn(1, 1, 4, 4))
    assert not model[0]._forward_hooks


def test_recommendations_are_structural():
    assert "gradcam" in [r.method for r in Inspector(BarNet()).suggest()]
    assert "gradcam" not in [r.method for r in Inspector(nn.Linear(2, 2)).suggest()]
    with pytest.raises(TypeError):
        Inspector(object())


def test_checkpoint(tmp_path):
    original = BarNet()
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": original.state_dict()}, path)
    restored = Inspector(BarNet(), checkpoint=path).model
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in original.state_dict().items())


def test_direction_and_degenerate_validation():
    assert torch.equal(concept_direction(torch.tensor([[2., 0.]]), torch.zeros(1, 2)), torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match="zero"):
        concept_direction(torch.zeros(1, 2), torch.zeros(1, 2))


@pytest.mark.parametrize("axis,shape", [(1, (2, 3, 4, 4)), (-1, (2, 4, 3))])
def test_steering_axes_and_restoration(axis, shape):
    model = nn.Sequential(nn.Identity())
    x = torch.zeros(shape)
    direction = torch.tensor([1., 0., 0.])
    with steer(model, "0", direction, feature_axis=axis, strength=2):
        y = model(x)
    assert torch.all(y.select(axis, 0) == 2)
    assert torch.equal(model(x), x)
    assert not model[0]._forward_hooks


def test_projection_formula_and_exception_cleanup():
    model = nn.Sequential(nn.Identity())
    x = torch.tensor([[2., 3.], [-2., 3.]])
    with steer(model, "0", torch.tensor([1., 0.]), mode="remove_positive_projection", strength=2.5):
        assert torch.equal(model(x), torch.tensor([[-3., 3.], [-2., 3.]]))
    with pytest.raises(RuntimeError):
        with steer(model, "0", torch.ones(2)):
            raise RuntimeError("forward failure")
    assert not model[0]._forward_hooks


def test_bad_steering_shape_cleanup():
    model = nn.Sequential(nn.Identity())
    with pytest.raises(ValueError, match="width"):
        with steer(model, "0", torch.ones(3)):
            model(torch.ones(2, 2))
    assert not model[0]._forward_hooks
