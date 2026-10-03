"""Small randomly initialized CPU teaching models, never pretrained models."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class TinyUNet(nn.Module):
    """One down/up stage with a skip concatenation; [B,1,H,W] -> [B,2,H,W]."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(12, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        skip = self.encoder(x)
        low = self.bottleneck(F.avg_pool2d(skip, 2))
        up = F.interpolate(low, size=skip.shape[-2:], mode="nearest")
        return self.head(self.decoder(torch.cat((skip, up), dim=1)))


class TinyTransformer(nn.Module):
    """Bidirectional fixed-length classifier; token IDs [B,T], scores [B,2].

    No causal mask or padding support. Embedding inputs are [B,T,width].
    """
    def __init__(self, vocab_size=12, width=8, max_length=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.randn(1, max_length, width) * 0.02)
        self.block = nn.TransformerEncoderLayer(width, 2, dim_feedforward=16,
                                                dropout=0.0, batch_first=True)
        self.head = nn.Linear(width, 2)

    def forward_embeddings(self, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[1] > self.position.shape[1]:
            raise ValueError("embeddings require [B,T,width] with T <= max_length")
        hidden = self.block(embeddings + self.position[:, :embeddings.shape[1]])
        return self.head(hidden.mean(1))

    def forward(self, token_ids):
        return self.forward_embeddings(self.embedding(token_ids))


@contextmanager
def patch_token(module: nn.Module, donor: torch.Tensor, token: int):
    """Replace one [B,T,D] output position by a detached matched donor.

    This is a local intervention, not a causal proof. Batch and shape must match;
    all examples receive their corresponding donor token. Hook always removed.
    """
    if donor.ndim != 3 or not 0 <= token < donor.shape[1]:
        raise ValueError("donor must be [B,T,D] and token in range")
    saved = donor.detach().clone()

    def patch(_module, _args, output):
        if not isinstance(output, torch.Tensor) or output.shape != saved.shape:
            raise ValueError("recipient and donor shapes must match")
        result = output.clone()
        result[:, token] = saved[:, token].to(output)
        return result

    handle = module.register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()


class TinyDenoiser(nn.Module):
    """2D noise predictor epsilon(x_t,t); pedagogical DDPM-style toy only."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(3, 16), nn.Tanh())
        self.head = nn.Linear(16, 2)

    def forward(self, x, timestep):
        t = torch.as_tensor(timestep, dtype=x.dtype, device=x.device)
        t = t.expand(x.shape[0]).reshape(-1, 1)
        return self.head(self.hidden(torch.cat((x, t), dim=1)))


def diffusion_sample(model, initial, *, steps=12):
    """Deterministic DDIM-style eta=0 trajectory, no final stochastic noise.

    Linear beta schedule [0.01,0.08]; training must use the same schedule.
    Caller owns RNG, evaluation mode and any steering hook scope.
    """
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    alpha = (1 - torch.linspace(0.01, 0.08, steps, device=initial.device,
                              dtype=initial.dtype)).cumprod(0)
    x = initial.clone()
    with torch.no_grad():
        for index in range(steps - 1, -1, -1):
            noise = model(x, index / (steps - 1))
            clean = (x - (1 - alpha[index]).sqrt() * noise) / alpha[index].sqrt()
            previous = alpha[index - 1] if index else x.new_tensor(1.0)
            x = previous.sqrt() * clean + (1 - previous).sqrt() * noise
    return x
