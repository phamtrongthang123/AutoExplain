import numpy as np
import pytest
import torch
from torch import nn

from autoexplain.backends import CAPTUM_METHODS, CAM_METHODS, LAYER_METHODS, captum_attribute, cam_attribute, captum_layer, shap_explain


def model():
    torch.manual_seed(11)
    return nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))


@pytest.mark.parametrize('method', CAPTUM_METHODS)
def test_captum_methods(method):
    pytest.importorskip('captum')
    torch.set_num_threads(2)
    net = model()
    x = torch.tensor([[0.2, 0.4, 0.8], [0.6, 0.3, 0.1]])
    kw = {}
    if method in {'gradient_shap', 'deeplift_shap'}:
        kw['baseline'] = torch.stack([torch.zeros(3), torch.ones(3)*0.05])
    if method == 'occlusion':
        kw['sliding_window_shapes'] = (1,)
    if method in {'lime','kernel_shap'}:
        kw['n_samples'] = 32
    if method in {'shapley_sampling','gradient_shap'}:
        kw['n_samples'] = 8
    if method == 'integrated_gradients':
        kw['n_steps'] = 8
    result = captum_attribute(net, x, method=method, target=1, **kw)
    assert result.shape == x.shape and result.isfinite().all()
    assert net.training


@pytest.mark.parametrize('method', CAM_METHODS)
def test_cam_methods(method):
    pytest.importorskip('pytorch_grad_cam')
    torch.set_num_threads(2)
    net = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    net[0].weight.grad = torch.ones_like(net[0].weight)
    result = cam_attribute(net, torch.rand(1, 1, 8, 8), layers=['1'], method=method, target=0)
    assert result.shape == (1,8,8) and result.isfinite().all()
    assert torch.equal(net[0].weight.grad, torch.ones_like(net[0].weight))
    assert net.training and not net[1]._forward_hooks


@pytest.mark.parametrize('method', LAYER_METHODS)
def test_captum_layers(method):
    pytest.importorskip('captum')
    result = captum_layer(model(), torch.ones(2,3), layer='1', method=method, target=0)
    assert result.shape == (2,5) and result.isfinite().all()


@pytest.mark.parametrize('method',['tree','permutation','exact','linear'])
def test_shap(method):
    pytest.importorskip('shap')
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LinearRegression
    x = np.random.default_rng(7).normal(size=(32,3))
    y = (x[:,0] > 0).astype(int)
    estimator = DecisionTreeClassifier(max_depth=2).fit(x,y)
    obj = estimator if method == 'tree' else estimator.predict_proba
    if method == 'linear':
        obj = LinearRegression().fit(x,y)
    result = shap_explain(obj, x[:2], method=method, background=x[:8])
    assert result.values.shape[:2] == (2,3) and np.isfinite(result.values).all()


def test_budgets():
    with pytest.raises(ValueError,match='six'):
        captum_attribute(nn.Linear(10,2),torch.ones(1,10),method='shapley_exact',target=0)
