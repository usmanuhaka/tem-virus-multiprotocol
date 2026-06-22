"""
TEM Virus Classifier — Streamlit demo
=====================================

Single-model inference demo for the DenseNet201 + TTA + mixup classifier from
"Multi-Protocol Re-Evaluation of TEM Virus Image Classification with
Source-Aware Splits and Ensemble Learning."

Inference is unchanged from the validated baseline: a timm DenseNet201 (14
classes), 224x224 input, ImageNet normalization, and 4-view test-time
augmentation (original, horizontal flip, vertical flip, both) with the softmax
averaged across the four views.

Weights (best.pt, ~74 MB) are not bundled; the app resolves them from the
WEIGHTS_URL / WEIGHTS_PATH secrets (or environment variables) and downloads the
release asset on first run. Torch/timm are imported lazily so the interface can
render before the model is touched.
"""

import io
import os
import math
import base64
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration — must match the training pipeline
# --------------------------------------------------------------------------- #
CLASS_NAMES = [
    "Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza",
    "Lassa", "Marburg", "Nipah virus", "Norovirus", "Orf", "Papilloma",
    "Rift Valley", "Rotavirus",
]
NUM_CLASSES = len(CLASS_NAMES)            # 14
IMG_SIZE = 224
TIMM_MODEL_NAME = "densenet201"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reported test metrics for the shipped checkpoint (Protocol A, seed 42).
TEST_ACCURACY = "96.4%"
TEST_MACRO_F1 = "0.957"
CHECKPOINT_EPOCH = "34"

REPO_URL = "https://github.com/usmanuhaka/tem-virus-multiprotocol"
DATASET_DOI = "https://doi.org/10.1016/j.cmpb.2021.106318"

# Palette (kept in sync with style.css, for inline SVG that cannot read vars).
INK = "#16211f"
INK_FAINT = "#6c7a77"
TRACK = "#edf1f0"
ACCENT = "#0f6e66"

ACCEPTED_TYPES = ["tif", "tiff", "png", "jpg", "jpeg"]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def load_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    try:
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    except OSError:
        pass


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def confidence_ring(pct: float) -> str:
    frac = max(0.0, min(1.0, pct / 100.0))
    r = 42.0
    circ = 2 * math.pi * r
    off = circ * (1 - frac)
    return (
        f'<svg class="ring" width="96" height="96" viewBox="0 0 96 96" role="img" '
        f'aria-label="confidence {pct:.0f} percent">'
        f'<circle cx="48" cy="48" r="42" fill="none" stroke="{TRACK}" stroke-width="9"/>'
        f'<circle cx="48" cy="48" r="42" fill="none" stroke="{ACCENT}" stroke-width="9" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{off:.2f}" '
        f'transform="rotate(-90 48 48)"/>'
        f'<text x="48" y="45" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="IBM Plex Mono, monospace" font-size="19" font-weight="600" fill="{INK}">{pct:.0f}%</text>'
        f'<text x="48" y="61" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="IBM Plex Mono, monospace" font-size="8.5" letter-spacing="1" fill="{INK_FAINT}">CONF</text>'
        f'</svg>'
    )


def _bars_html(probs: np.ndarray, idxs, top_idx: int) -> str:
    rows = []
    for i in idxs:
        pct = float(probs[i]) * 100
        cls = "bar-row is-top" if i == top_idx else "bar-row"
        rows.append(
            f'<div class="{cls}">'
            f'<div class="bar-name">{CLASS_NAMES[i]}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bar-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


# --------------------------------------------------------------------------- #
# UI blocks
# --------------------------------------------------------------------------- #
def render_header() -> None:
    chips = [
        ("Model", "DenseNet201"), ("Classes", "14"), ("Input", "224×224"),
        ("Inference", "4-view TTA"), ("Test acc", TEST_ACCURACY),
    ]
    chip_html = "".join(f'<span class="chip">{k}&nbsp;<b>{v}</b></span>' for k, v in chips)
    st.markdown(
        f'<div class="app-header">'
        f'<div class="app-rule"></div>'
        f'<h1 class="app-title">TEM Virus Classifier</h1>'
        f'<p class="app-subtitle">Deep-learning identification of viruses in transmission '
        f'electron microscopy images across 14 species — a research demonstration of the '
        f'DenseNet201 model from the multi-protocol re-evaluation study.</p>'
        f'<div class="chips">{chip_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_specimen(image: Image.Image | None = None, name: str | None = None) -> None:
    if image is None:
        st.markdown(
            '<div class="card"><div class="card-title">Specimen</div>'
            '<div class="empty">'
            '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#6c7a77" '
            'stroke-width="1.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/>'
            '<line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>'
            '<div class="empty-t">No specimen loaded — upload a TEM crop to preview it here.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return
    w, h = image.size
    st.markdown(
        f'<div class="card"><div class="card-title">Specimen</div>'
        f'<figure class="specimen">'
        f'<div class="specimen-frame"><img src="{_b64_png(image)}" alt="uploaded specimen"/></div>'
        f'<figcaption class="figcaption"><span><b>{name or "image"}</b></span>'
        f'<span>{w}×{h} px</span></figcaption>'
        f'</figure></div>',
        unsafe_allow_html=True,
    )


def render_empty_result() -> None:
    st.markdown(
        '<div class="card"><div class="card-title">Prediction</div>'
        '<div class="empty">'
        '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#6c7a77" '
        'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 12h4l3 8 4-16 3 8h4"/></svg>'
        '<div class="empty-t">Awaiting a specimen — class probabilities will appear here after upload.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_unavailable(detail: str | None = None) -> None:
    extra = f'<div class="figcaption" style="justify-content:flex-start">{detail}</div>' if detail else ""
    st.markdown(
        '<div class="card"><div class="card-title">Prediction</div>'
        '<div class="empty">'
        '<div class="empty-t">Model weights are not available to this deployment. '
        'Set the <code>WEIGHTS_URL</code> secret to the release asset and redeploy.</div>'
        f'</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def render_prediction(probs: np.ndarray, top_k: int = 5) -> None:
    order = list(np.argsort(probs)[::-1])
    top = order[0]
    pct = float(probs[top]) * 100
    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">Prediction <span class="count">mean softmax · 4-view TTA</span></div>'
        f'<div class="pred">{confidence_ring(pct)}'
        f'<div class="pred-meta">'
        f'<div class="pred-kicker">Most likely class</div>'
        f'<div class="pred-name">{CLASS_NAMES[top]}</div>'
        f'<div class="pred-note">Confidence is the averaged softmax probability over four flip views.</div>'
        f'</div></div>'
        f'<hr class="hairline"/>'
        f'<div class="card-title" style="margin-bottom:.85rem">Class probabilities '
        f'<span class="count">top {top_k}</span></div>'
        f'{_bars_html(probs, order[:top_k], top)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_model_card(device_label: str = "—") -> None:
    rows = [
        ("Architecture", "DenseNet201 (timm)", False),
        ("Parameters", "≈ 20 M", False),
        ("Input", "224×224 · ImageNet", False),
        ("Training", "mixup · warm restarts", False),
        ("Inference", "4-view TTA", False),
        ("Protocol", "A · seed 42", False),
        ("Checkpoint", f"epoch {CHECKPOINT_EPOCH}", False),
        ("Test accuracy", TEST_ACCURACY, True),
        ("Macro F1", TEST_MACRO_F1, True),
        ("Compute", device_label, False),
    ]
    body = "".join(
        f'<div class="meta-row"><span class="meta-key">{k}</span>'
        f'<span class="meta-val{" accent" if acc else ""}">{v}</span></div>'
        for k, v, acc in rows
    )
    st.markdown(
        f'<div class="card"><div class="card-title">Model</div>'
        f'<div class="meta">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_supported_classes() -> None:
    chips = "".join(
        f'<span class="classchip"><span class="idx">{i + 1:02d}</span>{name}</span>'
        for i, name in enumerate(CLASS_NAMES)
    )
    st.markdown(
        f'<div class="card"><div class="card-title">Supported classes '
        f'<span class="count">14</span></div>'
        f'<div class="classgrid">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        '<div class="note"><div class="note-title">Research demonstration — not for diagnosis</div>'
        '<p>This tool is a research demonstration, not a medical device, and must not inform '
        'clinical or diagnostic decisions. The model only recognizes the 14 virus classes in its '
        'training corpus; out-of-distribution inputs — other species, multi-particle fields, scale '
        'bars, or annotations — are still forced into one of these classes and will be unreliable.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f'<div class="footer"><p>'
        f'Model and code from the multi-protocol re-evaluation study (under review). '
        f'Images from the public TEM virus corpus of Matuszewski &amp; Sintorn (2021).'
        f'<br><a href="{REPO_URL}" target="_blank">Repository</a><span class="sep">/</span>'
        f'<a href="{DATASET_DOI}" target="_blank">Dataset (CMPB 2021)</a>'
        f'</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model & inference (torch imported lazily)
# --------------------------------------------------------------------------- #
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint.items()}


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    import torch
    import timm
    model = timm.create_model(TIMM_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(_extract_state_dict(ckpt), strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return model


def device_label(model) -> str:
    try:
        return "GPU (CUDA)" if next(model.parameters()).device.type == "cuda" else "CPU"
    except Exception:  # noqa: BLE001
        return "—"


def predict(model, image: Image.Image) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    device = next(model.parameters()).device
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
    probs = torch.zeros(1, NUM_CLASSES, device=device)
    with torch.no_grad():
        for view in views:
            probs += F.softmax(model(view), dim=1)
    probs /= 4.0
    return probs.squeeze(0).cpu().numpy()


def _secret(name: str, default: str = "") -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001
        pass
    return default


def resolve_weights():
    return _secret("WEIGHTS_PATH", "weights/best.pt"), _secret("WEIGHTS_URL", "")


def _download_weights(url: str, dest: str) -> None:
    p = Path(dest)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, p)


def get_model():
    """Resolve, download (if needed) and load the model. Returns (model, label, error)."""
    path, url = resolve_weights()
    try:
        if url and not Path(path).exists():
            with st.spinner("Fetching model weights…"):
                _download_weights(url, path)
        if not Path(path).exists():
            return None, "—", "weights-missing"
        with st.spinner("Loading model…"):
            model = load_model(path)
        return model, device_label(model), None
    except Exception as exc:  # noqa: BLE001
        return None, "—", str(exc)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="TEM Virus Classifier",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_css()
    render_header()

    model, dev_label, err = get_model()

    st.markdown(
        '<div class="eyebrow">Input image</div>'
        '<p style="margin:-.15rem 0 .55rem 0;color:#6c7a77;font-size:.9rem">'
        'Drag a TEM micrograph crop — a single particle works best. '
        'Grayscale or RGB; TIF, PNG, or JPG.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a TEM crop", type=ACCEPTED_TYPES, label_visibility="collapsed",
    )

    image = None
    if uploaded is not None:
        try:
            image = Image.open(io.BytesIO(uploaded.read()))
        except Exception:  # noqa: BLE001
            st.error("That file could not be read as an image. Try a TIF, PNG, or JPG.")

    probs = None
    if image is not None and model is not None:
        probs = predict(model, image)

    left, right = st.columns(2, gap="large")
    with left:
        render_specimen(image, uploaded.name if uploaded else None)
    with right:
        if image is None:
            render_empty_result()
        elif model is None:
            render_unavailable(None if err == "weights-missing" else f"Loader error: {err}")
        else:
            render_prediction(probs, top_k=5)

    if probs is not None:
        with st.expander("All 14 class probabilities"):
            order = list(np.argsort(probs)[::-1])
            st.markdown(_bars_html(probs, order, order[0]), unsafe_allow_html=True)

    meta_l, meta_r = st.columns(2, gap="large")
    with meta_l:
        render_model_card(dev_label)
    with meta_r:
        render_supported_classes()

    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()
