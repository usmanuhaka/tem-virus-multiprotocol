"""
TEM Virus Classifier - Streamlit demo
=====================================

Single-model inference demo for the DenseNet201 + TTA + mixup classifier from:

    "Multi-Protocol Re-Evaluation of TEM Virus Image Classification with
     Source-Aware Splits and Ensemble Learning"

The model is a timm DenseNet201 (14 classes) trained on the Matuszewski & Sintorn
TEM virus corpus, Protocol A. Inference reproduces the paper's evaluation setting:
224x224 input, ImageNet normalization, and 4-view test-time augmentation
(original, horizontal flip, vertical flip, and both), with softmax averaged
across the four views.

The trained weights (best.pt, ~74 MB) are NOT bundled in this repository.
See app/README.md for how to obtain them. By default the app looks for the
file at the path set in the sidebar (default: ../weights/best.pt) or at the URL
in the WEIGHTS_URL environment variable.
"""

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as T

# --------------------------------------------------------------------------- #
# Configuration  (must match the training pipeline)
# --------------------------------------------------------------------------- #
# Class order is the label_id order from the split manifests (alphabetical).
CLASS_NAMES = [
    "Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza",
    "Lassa", "Marburg", "Nipah virus", "Norovirus", "Orf", "Papilloma",
    "Rift Valley", "Rotavirus",
]
NUM_CLASSES = len(CLASS_NAMES)          # 14
IMG_SIZE = 224
TIMM_MODEL_NAME = "densenet201"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "../weights/best.pt")
WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation transform: deterministic resize + normalize (no random crop).
_eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _extract_state_dict(checkpoint):
    """Return a plain state_dict from various checkpoint layouts."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    # strip a leading "module." (DataParallel) if present
    return { (k[7:] if k.startswith("module.") else k): v
             for k, v in checkpoint.items() }


@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    """Build DenseNet201 (14 classes) and load trained weights. Cached."""
    model = timm.create_model(TIMM_MODEL_NAME, pretrained=False,
                              num_classes=NUM_CLASSES)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, missing, unexpected


def _maybe_download_weights(url: str, dest: str) -> str:
    """Download weights to `dest` if not already present. Returns the path."""
    dest_p = Path(dest)
    if dest_p.exists():
        return str(dest_p)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading weights from {url} ..."):
        urllib.request.urlretrieve(url, dest_p)
    return str(dest_p)


# --------------------------------------------------------------------------- #
# Inference with 4-view TTA
# --------------------------------------------------------------------------- #
def _tta_views(x: torch.Tensor):
    """original, hflip, vflip, hflip+vflip  (dims: N,C,H,W)."""
    return [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[2]),
        torch.flip(x, dims=[2, 3]),
    ]


@torch.no_grad()
def predict(model, image: Image.Image) -> np.ndarray:
    """Return a length-14 softmax vector averaged over the 4 TTA views."""
    x = _eval_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.zeros(1, NUM_CLASSES, device=DEVICE)
    for view in _tta_views(x):
        probs += F.softmax(model(view), dim=1)
    probs /= 4.0
    return probs.squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="TEM Virus Classifier", page_icon="🦠",
                   layout="centered")

st.title("🦠 TEM Virus Classifier")
st.caption(
    "DenseNet201 + TTA + mixup · 14 virus classes · Protocol A · "
    "test accuracy ≈ 0.97 (single model)."
)

with st.sidebar:
    st.header("Model")
    weights_path = st.text_input("Weights file (best.pt)", DEFAULT_WEIGHTS_PATH)
    st.markdown(
        "Weights are **not** shipped in the repo (~74 MB). "
        "Download them (see `app/README.md`) and point to the file here, "
        "or set the `WEIGHTS_URL` environment variable to auto-download."
    )
    st.markdown(f"**Device:** `{DEVICE}`")
    st.divider()
    top_k = st.slider("Show top-K classes", 3, NUM_CLASSES, 5)

# Resolve weights (optional auto-download)
resolved_path = weights_path
if WEIGHTS_URL and not Path(weights_path).exists():
    try:
        resolved_path = _maybe_download_weights(WEIGHTS_URL, weights_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not download weights: {exc}")

if not Path(resolved_path).exists():
    st.warning(
        f"Weights not found at `{resolved_path}`. "
        "Set the correct path in the sidebar, then re-run."
    )
    st.stop()

try:
    model, missing, unexpected = load_model(resolved_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load the model: {exc}")
    st.stop()

if missing or unexpected:
    st.info(
        f"Loaded with {len(missing)} missing and {len(unexpected)} unexpected "
        "keys (usually harmless if the architecture matches)."
    )

uploaded = st.file_uploader(
    "Upload a TEM crop (.tif, .png, .jpg)",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)

    probs = predict(model, image)
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])

    with col_pred:
        st.metric("Predicted class", CLASS_NAMES[pred_idx],
                  f"{probs[pred_idx] * 100:.1f}% confidence")

    df = pd.DataFrame({
        "class": [CLASS_NAMES[i] for i in order[:top_k]],
        "probability": [float(probs[i]) for i in order[:top_k]],
    }).set_index("class")
    st.subheader(f"Top-{top_k} predictions")
    st.bar_chart(df)
    st.dataframe(
        df.style.format({"probability": "{:.4f}"}),
        use_container_width=True,
    )

    st.caption(
        "Inference uses 4-view test-time augmentation (original + horizontal "
        "+ vertical + both flips), softmax-averaged — matching the paper."
    )
else:
    st.info("Upload a TEM image to run classification.")

st.divider()
st.caption(
    "Demo model: single DenseNet201 (Protocol A, seed 42). The paper's best "
    "configuration is a DenseNet201 + EfficientNetV2-S softmax ensemble; an "
    "ensemble demo additionally requires the EfficientNetV2-S weights."
)
