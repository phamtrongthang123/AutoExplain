import pytest
import torch
from autoexplain.toys import TinyUNet, TinyTransformer, TinyDenoiser, diffusion_sample, patch_token


def test_unet_odd_spatial_shape():
    model = TinyUNet()
    assert model(torch.zeros(2, 1, 9, 11)).shape == (2, 2, 9, 11)


def test_transformer_embedding_gradient():
    model = TinyTransformer().eval()
    tokens = torch.zeros(2, 4, dtype=torch.long)
    embedded = model.embedding(tokens).detach().requires_grad_(True)
    scores = model.forward_embeddings(embedded)
    assert scores.shape == (2, 2)
    gradient = torch.autograd.grad(scores[:, 0].sum(), embedded)[0]
    assert gradient.shape == (2, 4, 8)
    assert torch.isfinite(gradient).all()


def test_patch_exact_and_cleanup():
    module = torch.nn.Identity()
    recipient = torch.zeros(2, 3, 4)
    donor = torch.ones_like(recipient)
    with patch_token(module, donor, 1):
        output = module(recipient)
        torch.testing.assert_close(output[:, 1], donor[:, 1])
        assert torch.count_nonzero(output[:, 0]) == 0
    torch.testing.assert_close(module(recipient), recipient)
    with pytest.raises(RuntimeError):
        with patch_token(module, donor, 1):
            raise RuntimeError("cleanup")
    assert not module._forward_hooks


def test_patch_shape_validation():
    module = torch.nn.Identity()
    with pytest.raises(ValueError):
        with patch_token(module, torch.ones(1, 3, 4), 0):
            module(torch.ones(2, 3, 4))
    assert not module._forward_hooks


def test_diffusion_deterministic_and_no_input_mutation():
    model = TinyDenoiser().eval()
    initial = torch.ones(3, 2)
    first = diffusion_sample(model, initial)
    torch.testing.assert_close(first, diffusion_sample(model, initial))
    torch.testing.assert_close(initial, torch.ones_like(initial))
    assert first.shape == initial.shape and torch.isfinite(first).all()
    with pytest.raises(ValueError):
        diffusion_sample(model, initial, steps=1)
