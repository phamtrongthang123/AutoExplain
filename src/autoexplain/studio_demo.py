"""Pretrained ImageNet CNN explanations for user-uploaded photographs."""
import hashlib
import io
from importlib import resources
import os
from pathlib import Path
import tempfile
import time
import urllib.request


def unload():
    import streamlit as st
    st.session_state.pop("cnn_model", None)


def _weights_path():
    import torch
    return Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"


def _verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if not digest.hexdigest().startswith("f37072fd"):
        raise ValueError("Cached weights failed the official checksum. Remove this checkpoint with your cache manager before retrying.")


def _prepare(consent):
    import streamlit as st
    import torch
    from torchvision.models import resnet18
    from . import studio_jobs

    path = _weights_path()
    if not path.is_file():
        if not consent:
            raise ValueError("Allow the official weight download, or prepare the checkpoint in the Torch cache first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                progress = st.empty()
                started = time.monotonic()
                with urllib.request.urlopen("https://download.pytorch.org/models/resnet18-f37072fd.pth", timeout=20) as source:
                    total = 0
                    while True:
                        if time.monotonic() - started > 180:
                            raise ValueError("Download exceeded its three-minute budget. Retry explicitly when the connection is available.")
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > 64 * 1024 * 1024:
                            raise ValueError("Official checkpoint exceeded the 64 MiB transfer limit.")
                        output.write(block)
                        studio_jobs.stage("cnn", "Downloading official ImageNet weights")
                        progress.info(f"Downloading official weights: {total / 1024**2:.1f} MiB · Stop cancels between chunks")
            _verify(temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _verify(path)
    studio_jobs.stage("cnn", "Loading verified pretrained weights on CPU")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _image(content):
    from PIL import Image, ImageOps
    if not content or len(content) > 10 * 1024 * 1024:
        raise ValueError("Choose a nonempty photograph of at most 10 MiB.")
    with Image.open(io.BytesIO(content)) as image:
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("Choose a PNG, JPEG or WebP image.")
        if max(image.size) > 4096 or image.width * image.height > 16000000:
            raise ValueError("Image limit: 4096 pixels per side and 16 million pixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    import matplotlib.pyplot as plt
    import streamlit as st
    import torch
    try:
        from torchvision.models import ResNet18_Weights
    except (ImportError, RuntimeError):
        st.error("This workflow requires torchvision matched to your PyTorch installation. Prepare a compatible environment; no package is installed automatically.")
        return
    from autoexplain import Inspector
    from . import studio_jobs

    weights = ResNet18_Weights.IMAGENET1K_V1
    labels = weights.meta["categories"]
    st.title("Real CNN images")
    st.caption("ImageNet-pretrained ResNet-18 · your photograph · CPU inference · no training or synthetic inputs")
    st.write("Try the bundled real photograph or upload your own, prepare the pretrained classifier, then explicitly run predictions and explanations.")
    available = _weights_path().is_file()
    st.write("**Ready**" if "cnn_model" in st.session_state else "**Cached · prepare to load**" if available else "**Weights not cached**")
    st.caption("Official torchvision ImageNet-1K V1 weights, about 45 MiB; 1,000 ImageNet classes. CPU-only here; allow roughly 1 GiB of free RAM. Cache presence is not checksum validation until Prepare.")
    consent = st.checkbox("Allow downloading the official pretrained ResNet-18 weights if missing", key="cnn_download_consent")
    prepare, release = st.columns(2)
    if prepare.button("Prepare pretrained CNN", disabled=not available and not consent, type="primary"):
        try:
            with studio_jobs.task("cnn", "Prepare pretrained CNN"):
                st.session_state.cnn_model = _prepare(consent)
            st.rerun()
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else "Preparation failed. Check network/cache access and installed torchvision; no fallback model was used."
            studio_jobs.failure("cnn", message)
            st.error(message)
    if release.button("Unload CNN", disabled="cnn_model" not in st.session_state):
        unload()
        st.rerun()
    source = st.radio("Image source", ["Example photograph · NASA astronaut", "Upload my photograph"], key="cnn_image_source")
    if source == "Example photograph · NASA astronaut":
        content = resources.files("autoexplain").joinpath("_examples", "astronaut.png").read_bytes()
        st.caption("Real NASA photograph of Eileen Collins, bundled from scikit-image's public-domain astronaut sample. [Original source](https://flic.kr/p/r9qvLn). Not generated imagery or a recorded model prediction; ImageNet labels need not describe it correctly.")
    else:
        upload = st.file_uploader("Real photograph · PNG/JPEG/WebP · up to 10 MiB", type=["png", "jpg", "jpeg", "webp"], key="cnn_photo")
        content = upload.getvalue() if upload is not None and upload.size <= 10 * 1024 * 1024 else b""
        if upload is not None and not content:
            st.warning("Choose a nonempty image of at most 10 MiB.")
    if content:
        try:
            st.image(_image(content), caption="Original real photograph · model uses a center crop shown after the run", width=320)
        except Exception:
            st.error("Cannot preview this image. Choose a valid PNG/JPEG/WebP within the size limits.")
            content = b""
    choice = st.radio("Explain", ["Predicted class", "Choose a fixed ImageNet class"], horizontal=True)
    target = st.selectbox("Fixed ImageNet class", range(len(labels)), format_func=lambda i: f"{i}: {labels[i]}") if choice != "Predicted class" else None
    identity = (hashlib.sha256(content).hexdigest(), target)
    if st.button("Classify and explain photograph", type="primary", disabled=not content or "cnn_model" not in st.session_state):
        try:
            with studio_jobs.task("cnn", "Explain real photograph"):
                photo = _image(content)
                x = weights.transforms()(photo).unsqueeze(0)
                model = st.session_state.cnn_model
                studio_jobs.stage("cnn", "Computing ImageNet class scores")
                with torch.no_grad():
                    logits = model(x)[0]
                    probabilities = logits.softmax(-1)
                selected = int(logits.argmax()) if target is None else target
                inspector = Inspector(model)
                st.info("Computing fixed-class Grad-CAM · Stop takes effect between model forwards")
                studio_jobs.stage("cnn", "Grad-CAM at layer4")
                cam = inspector.gradcam(x, layer="layer4", target=selected)[0]
                st.info("Computing fixed-class input gradients")
                studio_jobs.stage("cnn", "Input gradient magnitude")
                gradient = inspector.input_gradients(x, target=selected)[0].abs().mean(0)
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                crop = (x[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                top = probabilities.topk(5)
                st.session_state.cnn_result = {
                    "identity": identity, "crop": crop, "cam": cam.numpy(), "gradient": gradient.numpy(),
                    "target": selected, "logit": float(logits[selected]), "probability": float(probabilities[selected]),
                    "top": [{"class": labels[i], "class_id": i, "probability": float(p), "logit": float(logits[i])}
                            for p, i in zip(top.values.tolist(), top.indices.tolist())]}
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else "Could not explain this image. Check image format, available RAM and model readiness; the previous result is retained."
            studio_jobs.failure("cnn", message)
            st.error(message)
    studio_jobs.render("cnn")
    saved = st.session_state.get("cnn_result")
    if saved:
        if saved["identity"] != identity:
            st.warning("Previous completed result; this is not an explanation of the current image/class selection.")
        st.subheader(f"Explained class: {labels[saved['target']]}")
        st.write({"class_id": saved["target"], "logit": saved["logit"], "ImageNet_softmax": saved["probability"]})
        st.dataframe(saved["top"], hide_index=True)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        try:
            axes[0].imshow(saved["crop"])
            axes[0].set_title("Actual model crop")
            axes[1].imshow(saved["crop"])
            axes[1].imshow(saved["cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            axes[1].set_title("Grad-CAM · layer4")
            axes[2].imshow(saved["gradient"], cmap="inferno")
            axes[2].set_title("Input gradient magnitude")
            for axis in axes:
                axis.axis("off")
            st.pyplot(fig)
        finally:
            plt.close(fig)
        st.caption("Preprocessing: resize shorter side to 256, center crop 224×224, ImageNet normalization. Grad-CAM shows positive, per-image-normalized evidence for the selected class logit. Input gradients show mean absolute channel sensitivity in normalized input coordinates, not signed contribution. Neither proves causality; ImageNet softmax is not calibrated confidence or medical validation.")
    st.caption("Nothing runs until you choose Prepare or Classify. Images/results remain in this browser session only; no image is uploaded to an external service or saved to disk. Navigation away unloads the CNN; cached weights and the last result remain. No synthetic fallback.")
