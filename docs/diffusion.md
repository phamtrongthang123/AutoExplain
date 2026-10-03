# Diffusion steering and PolypSteer

[PolypSteer](https://arxiv.org/abs/2603.07066) studies counterfactual endoscopic synthesis using training-free activation steering. Its [official code](https://github.com/UARK-AICV/PolypSteer) is the reference for reproducing the paper.

AutoExplain includes generic positive-projection removal and a small Diffusers sampling/intervention adapter (see [integrations](integrations.md) and notebook 11). It does not include a complete text-conditioned PolypSteer adapter. It does not download PixArt, estimate per-timestep directions, select classifier-free guidance branches, or generate endoscopic images.

## Runnable mathematical example

```python
import torch
from torch import nn
from autoexplain import steer

model = nn.Sequential(nn.Identity())
h = torch.tensor([[2., 3.], [-2., 3.]])
v = torch.tensor([1., 0.])
with steer(model, "0", v, strength=2.5,
           mode="remove_positive_projection"):
    changed = model(h)
print(changed)
# tensor([[-3., 3.], [-2., 3.]])
assert torch.equal(model(h), h)  # original behavior restored
```

The normalized direction `v` defines a feature axis. The operation is:

```text
h' = h - strength * max(sum(h * v), 0) * v
```

The sum is over the selected feature dimension. Positive projections are suppressed or reversed, depending on strength; negative projections are unchanged. For `[batch, tokens, features]`, use `feature_axis=-1`. For `[batch, channels, height, width]`, use `feature_axis=1`.

## What a full adapter must specify

The official PolypSteer implementation describes PixArt-α XL/2 at 512×512 with medical LoRA adaptation, matched-seed contrastive activations, conditional cross-attention outputs, and a layer window of indices 8–15. Reproduction must check the current official implementation and its assets, including timestep indexing and conditioning behavior. Training-free steering does not imply that the underlying model was never adapted.

A production diffusion adapter needs explicit handling of:

- Hook-site output structure and feature dimensions.
- Conditional versus unconditional branches under classifier-free guidance.
- Per-layer/per-timestep concept directions and matched reference conditions.
- Intervention timing, spatial selection policy and strength.
- Fixed prompts, initial noise and random-generator state for baseline comparisons.
- Safety, image quality and preservation of non-target content.

Do not apply the generic hook indiscriminately to a guidance batch and label the result a PolypSteer reproduction. The generic hook affects the entire selected tensor every time the module runs within its context.

## Small offline diffusion workflow

The [toy diffusion notebook](notebooks.md) trains a 2D noise predictor and compares
matched initial noise under zero, positive, negative and random hidden-direction
interventions. It uses a deterministic DDIM-style update and an intentionally
short schedule. The terminal noise distribution is approximate, and a direction
estimated at one timestep need not transfer across the trajectory. This example
teaches controls and hook placement; it is neither an image generator nor a
PolypSteer reproduction.

## Evaluation boundary

A prediction change alone is not a successful counterfactual. Evaluate targeted pathology changes, preservation outside the intended region, seed robustness and model failure cases. The synthetic playground provides no clinical validation, and AutoExplain is not a medical decision tool.
