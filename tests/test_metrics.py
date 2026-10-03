import pytest
import torch
from torch import nn

from autoexplain.metrics import perturbation_faithfulness


@pytest.mark.parametrize("mode", ["deletion", "insertion"])
def test_exact_curves_auc_random_controls_and_state(mode):
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3., 2., 1.]]))
    model.train()
    model.weight.grad = torch.ones_like(model.weight)
    x = torch.ones(1, 3)
    attrs = torch.tensor([[3., 2., 1.]])
    kwargs = dict(mode=mode, steps=3, random_trials=16)
    result = perturbation_faithfulness(model, x, attrs, generator=torch.Generator().manual_seed(7), **kwargs)
    expected = [6., 3., 1., 0.] if mode == "deletion" else [0., 3., 5., 6.]
    assert torch.equal(result.scores, torch.tensor([expected]))
    assert torch.allclose(result.auc, torch.tensor([7 / 3 if mode == "deletion" else 11 / 3]))
    assert result.random_scores.shape == (16, 1, 4)
    assert torch.equal(result.random_scores[:, :, 0], result.scores[:, 0].expand(16, 1))
    assert torch.equal(result.random_scores[:, :, -1], result.scores[:, -1].expand(16, 1))
    again = perturbation_faithfulness(model, x, attrs, generator=torch.Generator().manual_seed(7), **kwargs)
    assert torch.equal(result.random_scores, again.random_scores)
    if mode == "deletion":
        assert result.auc.item() < result.random_auc.mean().item()
    else:
        assert result.auc.item() > result.random_auc.mean().item()
    assert model.training and torch.equal(model.weight.grad, torch.ones_like(model.weight))


def test_shape_validation_and_failure_mode_restoration():
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="shape"):
        perturbation_faithfulness(model, torch.ones(1, 2), torch.ones(2))
    model.train()
    with pytest.raises(ValueError, match="target"):
        perturbation_faithfulness(model, torch.ones(1, 2), torch.ones(1, 2), target=4)
    assert model.training


def test_steps_capped_and_nonzero_baseline_endpoints():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    result = perturbation_faithfulness(model, torch.tensor([[2., 3.]]), torch.ones(1, 2),
                                     baseline=1., steps=100, random_trials=1)
    assert result.fractions.numel() == 3
    assert torch.equal(result.scores[:, [0, -1]], torch.tensor([[5., 2.]]))
