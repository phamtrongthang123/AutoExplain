import pytest
import torch
from torch import nn
from autoexplain.adapters import SegmentationScoreAdapter
from autoexplain.core import Inspector


def test_weighted_score_and_gradient():
    model = nn.Conv2d(1, 2, 1, bias=False)
    mask = torch.tensor([[1., 0.], [0., 0.]])
    wrapped = SegmentationScoreAdapter(model, mask)
    x = torch.ones(1, 1, 2, 2)
    torch.testing.assert_close(wrapped(x), model(x)[:, :, 0, 0])
    grad = Inspector(wrapped).input_gradients(x, target=0)
    assert grad.shape == x.shape
    assert torch.count_nonzero(grad[:, :, 1]) == 0
    assert grad[0, 0, 0, 1] == 0


@pytest.mark.parametrize("mask", [torch.zeros(2, 2), torch.ones(2), -torch.ones(2, 2), torch.full((2, 2), float("nan"))])
def test_invalid_mask(mask):
    with pytest.raises(ValueError):
        SegmentationScoreAdapter(nn.Identity(), mask)


def test_output_and_spatial_validation():
    with pytest.raises(ValueError):
        SegmentationScoreAdapter(nn.Identity())(torch.ones(2, 3))
    with pytest.raises(ValueError):
        SegmentationScoreAdapter(nn.Identity(), torch.ones(3, 3))(torch.ones(1, 2, 2, 2))
