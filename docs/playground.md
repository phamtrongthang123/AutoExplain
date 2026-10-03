# Runnable playground

The playground executes a small CNN, rather than displaying stored explanations. Run it locally:

```bash
git clone https://github.com/phamtrongthang123/AutoExplain.git
cd AutoExplain
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[playground]'
streamlit run playground/app.py
```

Open the localhost URL printed by Streamlit. First launch trains a small model; subsequent interactions reuse cached parameters. Models and hooks are local to each app execution, not shared between users.

## Try an explanation

Select a held-out example and choose **Vertical** or **Horizontal** as the explanation target. The app displays the input, a Grad-CAM overlay and a selectable attribution magnitude map. Choose input gradients, integrated gradients, SmoothGrad, occlusion or LayerCAM. Integrated gradients also reports its signed completeness residual; a small residual checks numerical integration, not semantic validity. Grad-CAM describes spatial features contributing positively to the selected class score. Input gradients show local score sensitivity, not a causal explanation.

Change the target while keeping the image fixed. If the maps look similar, that is a result to investigate rather than something the app suppresses. A zero map can mean there is no positive contribution at the selected layer; it does not necessarily mean the image contains no relevant information.

## Try an intervention

Move the steering slider. The direction is the normalized difference between vertical-bar and horizontal-bar mean feature activations, estimated on the reference training set. Held-out images use a different seed. The intervention adds that vector to the final feature map, then computes new class probabilities.

At strength zero, baseline and steered predictions should match. Positive strength moves features toward the vertical reference mean, but monotonic or semantically meaningful effects are not guaranteed in general. The input image does not change.

## Controls and donor patches

Enable deletion/insertion evaluation to compare the selected attribution ranking against three seeded random rankings. Scores are raw logits, the baseline is zero, and individual scalars are ranked by absolute magnitude. Charts show eight perturbation steps; the API integrates against actual changed fractions. AUC ordering can be misleading for negative evidence or out-of-distribution masks.

The donor-feature option replaces the recipient's complete feature tensor with another held-out example's features. The hook is scoped to one forward. A changed prediction demonstrates the intervention mechanism, not a discovered semantic circuit. Cached values are parameter dictionaries only; mutable hooked models are never shared.

For other model families, run the [offline notebook collection](notebooks.md). The app remains a focused CNN experiment rather than implying equivalent support for every architecture.

## Headless alternative

```bash
python -m autoexplain.demo
```

The command prints held-out accuracy, heatmap shape and finiteness, and a steering-induced logit difference. Synthetic accuracy is a demo check, not a benchmark for explanation quality.

## Hosting boundary

GitHub Pages serves static documentation and cannot execute PyTorch or host Streamlit. The documented app runs on your computer. A public browser-only live service would require a separate compute host; none is provisioned by this repository. Do not expose arbitrary checkpoint or Python uploads on a public server.
