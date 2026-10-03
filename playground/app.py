import matplotlib.pyplot as plt
import streamlit as st
import torch

from autoexplain import (Inspector, concept_direction, steer, integrated_gradients,
                         smoothgrad, occlusion, layercam, perturbation_faithfulness,
                         capture_activations, patch_activation)
from autoexplain.demo import bars, trained_demo

st.set_page_config(page_title="AutoExplain playground", layout="wide")
st.title("AutoExplain: explain, intervene, compare")
st.write("An actual CPU experiment, not a pre-rendered heatmap. The model learns vertical vs horizontal bars.")
torch.set_num_threads(2)

# Cache parameters, not a shared mutable model: hooks must be session-local.
@st.cache_data
def weights():
    return trained_demo().state_dict()

from autoexplain.demo import BarNet
model = BarNet()
model.load_state_dict(weights())
model.eval()
inspector = Inspector(model)
index = st.slider("Held-out example", 0, 63, 0)
target = st.selectbox("Explain class", [0, 1], format_func=lambda x: ["Vertical", "Horizontal"][x])
strength = st.slider("Vertical-concept steering strength", -2.0, 2.0, 0.0, 0.1)
images, labels = bars(64, seed=99)
x = images[index:index + 1]
method = st.selectbox("Attribution method", ["Input gradient", "Integrated gradients", "SmoothGrad", "Occlusion", "LayerCAM"])
cam = inspector.gradcam(x, target=target)[0]
if method == "Integrated gradients":
    result = integrated_gradients(model, x, target, steps=32)
    attribution = result.attributions
    st.metric("IG completeness residual (signed)", f"{result.completeness_delta.item():.5f}")
elif method == "SmoothGrad":
    attribution = smoothgrad(model, x, target, samples=12, generator=torch.Generator().manual_seed(13))
elif method == "Occlusion":
    attribution = occlusion(model, x, target, window_shape=(1, 4, 4), strides=(1, 4, 4))
elif method == "LayerCAM":
    attribution = layercam(model, x, target, layer="features.2")[:, None]
else:
    attribution = inspector.input_gradients(x, target=target)
gradient = attribution[0].abs().mean(0)
with torch.no_grad():
    baseline = model(x).softmax(-1)[0]
    reference, reference_labels = bars(128, seed=7)
    features = model.features(reference).mean((-2, -1))
    direction = concept_direction(features[reference_labels == 0], features[reference_labels == 1])
    with steer(model, "features", direction, feature_axis=1, strength=strength):
        changed = model(x).softmax(-1)[0]
    accuracy = (model(images).argmax(1) == labels).float().mean().item()

st.metric("Held-out synthetic accuracy", f"{accuracy:.1%}")
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for axis in axes:
    axis.axis("off")
axes[0].imshow(x[0, 0], cmap="gray")
axes[0].set_title(f"Input: {['vertical', 'horizontal'][labels[index]]}")
axes[1].imshow(x[0, 0], cmap="gray")
axes[1].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
axes[1].set_title("Grad-CAM")
axes[2].imshow(gradient, cmap="inferno")
axes[2].set_title(method + " (magnitude)")
st.pyplot(fig)
plt.close(fig)
st.table({"class": ["vertical", "horizontal"], "baseline probability": baseline.tolist(),
          "steered probability": changed.tolist()})
st.caption("Steering changes internal features, not the displayed input. Heatmaps are not proof of causality. "
           "This synthetic experiment is not a medical or diffusion-model validation.")
if st.checkbox("Evaluate deletion and insertion against random rankings"):
    for mode in ("deletion", "insertion"):
        curve = perturbation_faithfulness(model, x, attribution, target, mode=mode,
                                         steps=8, random_trials=3,
                                         generator=torch.Generator().manual_seed(42))
        st.write(f"{mode.title()} raw-logit AUC: {curve.auc.item():.4f}; "
                 f"random mean: {curve.random_auc.mean().item():.4f}")
        st.line_chart({"attribution ranking": curve.scores[0].tolist(),
                       "random mean": curve.random_scores[:, 0].mean(0).tolist()})
    st.caption("Horizontal axis is perturbation step (0–8); zero baseline, absolute scalar ranking. "
               "Negative evidence and out-of-distribution masking can reverse expected AUC ordering.")

if st.checkbox("Patch features from a held-out donor"):
    donor_index = st.slider("Donor example", 0, 63, 1)
    with torch.no_grad(), capture_activations(model, "features") as captured:
        model(images[donor_index:donor_index + 1])
    with torch.no_grad(), patch_activation(model, "features", captured["features"][0]):
        patched = model(x).softmax(-1)[0]
    st.table({"class": ["vertical", "horizontal"], "recipient": baseline.tolist(),
              "donor-feature patch": patched.tolist()})
    st.caption("Full feature replacement is a mechanical intervention, not evidence for a semantic circuit.")

with st.expander("Tutorials and research boundaries"):
    st.write("Thirteen offline CPU notebooks cover attribution libraries, SAE features, actual J-lens, "
             "OpenAI sparse-circuit code, Goodfire nano decomposition, real Diffusers image steering and "
             "circuit tracing alongside the original tutorials. Tracing uses a separate environment. "
             "See docs/integrations.md and docs/notebooks.md.")
    st.write("Established attribution baselines plus modern intervention workflows; no state-of-the-art "
             "performance claim. No pretrained BERT, JEPA, LLM or PolypSteer reproduction is bundled.")
with st.expander("Browse the tool collection"):
    from dataclasses import asdict
    from autoexplain.catalog import list_tools
    st.dataframe([asdict(tool) for tool in list_tools()], hide_index=True)
    st.caption("Operations include multiple backends for some algorithms. A listing is not proof of "
               "pretrained artifact support; read each scope field.")

if st.checkbox("Run an optional upstream CAM backend"):
    from autoexplain.backends import CAM_METHODS, cam_attribute
    backend_method = st.selectbox("Upstream CAM", list(CAM_METHODS))
    try:
        upstream_map = cam_attribute(model, x, layers=["features.3"], method=backend_method, target=target)
        fig, axis = plt.subplots(figsize=(3, 3))
        axis.imshow(upstream_map[0], cmap="inferno"); axis.axis("off")
        st.pyplot(fig); plt.close(fig)
        st.caption("EigenCAM is class-independent; methods have different assumptions.")
    except ImportError:
        st.warning("Install the vision extra to run this backend: pip install -e '.[vision]'")

with st.expander("Why these methods?"):
    for recommendation in inspector.suggest():
        st.write(recommendation.__dict__)
with st.expander("Run it on your model"):
    st.code('''from autoexplain import Inspector
inspector = Inspector(your_model, checkpoint="weights.pt")
print(inspector.suggest())
# Your preprocessing and class-score output adapter are required.
heatmap = inspector.gradcam(batch, layer="features.2", target=0)''')
