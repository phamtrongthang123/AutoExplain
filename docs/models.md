# Bring your model and checkpoint

Install the package from GitHub:

```bash
python -m pip install "autoexplain-ai @ git+https://github.com/phamtrongthang123/AutoExplain.git"
```

## Complete runnable checkpoint example

This example constructs a model, saves its state dict, reloads it and generates an explanation. Replace `BarNet` and the demo inputs with your own model definition and preprocessing.

```python
import tempfile
from pathlib import Path
import torch
from autoexplain import Inspector
from autoexplain.demo import BarNet, bars, trained_demo

torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "weights.pt"
    torch.save(trained_demo().state_dict(), checkpoint)
    inspector = Inspector(BarNet(), checkpoint=checkpoint)
    batch, labels = bars(4, seed=99)
    print(inspector.suggest())
    maps = inspector.gradcam(batch, layer="features.2", target=labels)
    print(maps.shape)  # torch.Size([4, 16, 16])
```

## Required information

You own the architecture definition, preprocessing, device placement, class meanings and output adaptation. The model must return a tensor of shape `[batch, classes]` for the explanation methods. Prefer logits over probabilities to avoid saturation. The default target is the highest-scoring class for each sample.

Grad-CAM chooses the last registered `Conv2d` when no layer is given. Registration order does not establish semantic relevance or execution order; explicitly selecting a verified layer is safer. The selected module must execute exactly once and return a four-dimensional activation connected to the selected score.

For dictionary, tuple or segmentation outputs, write an `nn.Module` wrapper that returns a clearly defined class-score tensor. For example, a segmentation score might average one class's logits within a fixed region. Defining that region changes the scientific question; AutoExplain does not silently choose it.

## Explicit segmentation scores

```python
from autoexplain.adapters import SegmentationScoreAdapter
from autoexplain import integrated_gradients

# segmentation_model returns [B,C,H,W]; roi is fixed nonnegative [H,W].
scorer = SegmentationScoreAdapter(segmentation_model, mask=roi)
result = integrated_gradients(scorer, images, target=1, steps=64)
print(result.completeness_delta)
```

The ROI defines weighted mean logits, not lesion probability or segmentation
quality. See the [U-Net notebook](notebooks.md) for a complete synthetic example.
For token IDs, differentiate floating embeddings instead of integer IDs; the tiny
transformer notebook makes that boundary explicit.

## Checkpoint safety

The loader calls `torch.load(..., weights_only=True)` and `load_state_dict(..., strict=True)`. It accepts a plain state dict or a dictionary containing `state_dict`. It does not execute a checkpoint's model-construction code or support arbitrary serialized Python objects. Restricted loading reduces risk but is not a sandbox; accept only trusted files and keep PyTorch updated.
