# app_streamlit.py
# Streamlit landing page for TraceFinder with sticker headings + PDF + description + prediction demo
# Run from ai_trace_finder folder (with venv activated):
#   python -m streamlit run app_streamlit.py

import streamlit as st
from pathlib import Path
import base64
import os
import sys

import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# ===== PyTorch CNN IMPORTS =====
import torch
import torch.nn.functional as F

# ===== Hybrid CNN (Keras) IMPORTS =====
import pickle
import tensorflow as tf

st.set_page_config(page_title="TraceFinder — Landing", layout="wide", page_icon="🟣")

# ------ CONFIG -------
BASE_DIR = Path(__file__).parent.resolve()

# Models folder (ai_trace_finder/models)
MODEL_DIR = BASE_DIR / "models"

# ===== Baseline paths (if available) =====
SCALER_PATH = MODEL_DIR / "scaler.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"
SVM_PATH = MODEL_DIR / "svm.pkl"

# ===== CNN PATH CONFIG =====
CNN_MODEL_PATH = MODEL_DIR / "cnn" / "cnn_model.pth"

# Add src to path and import CNN + hybrid utils
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cnn.model import SimpleCNN  # PyTorch CNN
from cnn.utils import process_batch_gpu, batch_corr_gpu, extract_enhanced_features

# Hybrid artifacts
RESULTS_DIR = BASE_DIR / "results" / "hybrid_cnn"
HYB_MODEL_PATH = RESULTS_DIR / "scanner_hybrid_final.keras"
HYB_ENCODER_PATH = RESULTS_DIR / "hybrid_label_encoder.pkl"
HYB_SCALER_PATH = RESULTS_DIR / "hybrid_feat_scaler.pkl"
HYB_FP_PATH = RESULTS_DIR / "scanner_fingerprints.pkl"
HYB_ORDER_NPY = RESULTS_DIR / "fp_keys.npy"

# ===== CNN CLASS LABELS (CURRENTLY PARTIAL, USED ONLY FOR TOP-1 LABEL) =====
CNN_CLASS_NAMES = [
    "Canon120-1",
    "Canon120-2",
    "Canon220",
    "Canon9000-1",
    "Canon9000-2",
    "EpsonV370-1",
    "EpsonV370-2",
    "EpsonV39-1",
    "EpsonV39-2",
    "EpsonV550",
    "HP",
    # remaining classes not listed; CNN still works for top-1 prediction
]

# Try local project PDF first, fallback to /mnt/data if available (useful for cloud)
DEFAULT_PDF_PATH = BASE_DIR / "AI_TraceFinder.pdf"
FALLBACK_PDF = Path("/mnt/data/AI_TraceFinder (1) (1).pdf")
if not DEFAULT_PDF_PATH.exists() and FALLBACK_PDF.exists():
    DEFAULT_PDF_PATH = FALLBACK_PDF

# Maximum bytes to safely embed as data URI (5 MB)
MAX_EMBED_BYTES = 5 * 1024 * 1024

# ------ HELPERS (PDF) -------
def load_pdf_bytes(path: Path):
    try:
        if path.exists() and path.is_file():
            return path.read_bytes(), path.name
    except Exception:
        pass
    return None, None


def pdf_to_data_uri(pdf_bytes: bytes) -> str:
    return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("utf-8")


# load default PDF
default_pdf_bytes, default_pdf_name = load_pdf_bytes(DEFAULT_PDF_PATH)

# ------ HELPERS (Prediction) -------
@st.cache_resource
def load_models():
    """Load RF/SVM baseline models if present."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")
    if not RF_PATH.exists():
        raise FileNotFoundError(f"Random Forest model not found at: {RF_PATH}")
    if not SVM_PATH.exists():
        raise FileNotFoundError(f"SVM model not found at: {SVM_PATH}")

    scaler = joblib.load(SCALER_PATH)
    rf = joblib.load(RF_PATH)
    svm = joblib.load(SVM_PATH)
    return scaler, rf, svm


# ===== CNN MODEL LOADER =====
@st.cache_resource
def load_cnn_model():
    """Load the trained PyTorch CNN model."""
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found at: {CNN_MODEL_PATH}")

    NUM_CLASSES = 22  # must match training config
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=NUM_CLASSES)
    state = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ===== HYBRID CNN LOADER =====
@st.cache_resource
def load_hybrid_artifacts():
    """Load hybrid Keras model + encoder + scaler + fingerprints."""
    if not HYB_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hybrid CNN model not found at: {HYB_MODEL_PATH}")
    if not HYB_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Hybrid label encoder not found at: {HYB_ENCODER_PATH}")
    if not HYB_SCALER_PATH.exists():
        raise FileNotFoundError(f"Hybrid feature scaler not found at: {HYB_SCALER_PATH}")
    if not HYB_FP_PATH.exists():
        raise FileNotFoundError(f"Scanner fingerprints not found at: {HYB_FP_PATH}")
    if not HYB_ORDER_NPY.exists():
        raise FileNotFoundError(f"Fingerprint key order not found at: {HYB_ORDER_NPY}")

    model = tf.keras.models.load_model(HYB_MODEL_PATH, compile=False)
    with open(HYB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(HYB_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(HYB_FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(str(HYB_ORDER_NPY), allow_pickle=True).tolist()

    return model, le, scaler, scanner_fps, fp_keys


def load_and_preprocess_from_bytes(file_bytes: bytes, size=(512, 512)):
    """Decode uploaded image bytes to grayscale, normalize and resize."""
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    img = img.astype(np.float32) / 255.0
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized


# ===== CNN PREPROCESSING =====
def preprocess_for_cnn(file_bytes: bytes, size=(128, 128)):
    """Prepare RGB image tensor for CNN (1 x 3 x H x W)."""
    file_array = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image for CNN.")

    img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # H,W,3

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return tensor


def compute_metadata_features_from_bytes(img, file_bytes: bytes):
    """Compute same metadata features used during training."""
    h, w = img.shape
    aspect_ratio = w / h if h != 0 else 0.0
    file_size_kb = len(file_bytes) / 1024.0 if file_bytes is not None else 0.0

    pixels = img.flatten()
    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))

    # histogram for entropy
    counts = np.histogram(pixels, bins=256, range=(0, 1))[0].astype(np.float64) + 1e-6
    ent = float(entropy(counts))

    edges = sobel(img)
    edge_density = float(np.mean(edges > 0.1))

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density,
    }


def run_prediction(uploaded_file, model_choice: str):
    """Run prediction for RF / SVM / CNN / Hybrid CNN and return (label, probs, class_names)."""
    file_bytes = uploaded_file.getvalue()

    # ================= Hybrid CNN =================
    if model_choice == "Hybrid CNN":
        hyb_model, le_hyb, scaler_hyb, scanner_fps, fp_keys = load_hybrid_artifacts()

        # Save to temp file path for process_batch_gpu
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            residuals = process_batch_gpu([tmp_path])
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if not residuals:
            raise ValueError("Hybrid pipeline could not compute residual for the image.")

        res = residuals[0].astype(np.float32)  # 256x256

        # PRNU + enhanced features
        corrs = batch_corr_gpu([res], scanner_fps, fp_keys)  # (1, K)
        enh = extract_enhanced_features(res)                 # (F,)
        feats_combined = np.hstack([corrs[0], enh])          # (K+F,)
        feats_scaled = scaler_hyb.transform([feats_combined])  # (1, K+F)

        X_img = np.expand_dims(res, axis=(0, -1))            # (1, 256, 256, 1)

        probs = hyb_model.predict([X_img, feats_scaled], verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        label = le_hyb.classes_[pred_idx]

        # return full probabilities and class names so UI can show table
        return label, probs, le_hyb.classes_

    # ================= CNN =================
    if model_choice == "CNN":
        model = load_cnn_model()
        x = preprocess_for_cnn(file_bytes)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        if pred_idx < len(CNN_CLASS_NAMES):
            label = CNN_CLASS_NAMES[pred_idx]
        else:
            label = f"Class-{pred_idx}"

        return label, None, None

    # ============== RF / SVM =================
    scaler, rf, svm = load_models()

    img = load_and_preprocess_from_bytes(file_bytes)
    features = compute_metadata_features_from_bytes(img, file_bytes)
    df_feat = pd.DataFrame([features])

    X_scaled = scaler.transform(df_feat)

    model = rf if model_choice == "Random Forest" else svm
    pred = model.predict(X_scaled)[0]

    probs = None
    class_names = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        class_names = model.classes_

    return pred, probs, class_names


# theme toggle state (unique key)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


# ===== Sidebar (New Feature) =====
with st.sidebar:
    st.markdown("### TraceFinder")
    st.caption("Scanner & Camera Source Identification — PRNU + Metadata")

    theme_choice = st.selectbox(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state.theme == "dark" else 1,
        key="theme_toggle_main_sticker",
    )
    st.session_state.theme = theme_choice

    st.markdown("---")
    st.markdown("**Quick Navigation**")
    st.markdown("- [Features](#features)")
    st.markdown("- [Use Cases](#use-cases)")
    st.markdown("- [Advantages](#advantages)")
    st.markdown("- [Prediction](#prediction)")
    st.markdown("- [Flow](#flow)")

    st.markdown("---")
    if default_pdf_bytes:
        pdf_uri = pdf_to_data_uri(default_pdf_bytes[:MAX_EMBED_BYTES])
        st.markdown(f'<a class="cta" href="{pdf_uri}" target="_blank">Open Project Brief (PDF)</a>', unsafe_allow_html=True)
    else:
        st.warning("Project PDF not found in project folder.")

    st.markdown("---")
    st.caption("Developed by Harsh Pandey")


# ------ CSS (Advanced UI + sticker badge + widget polish) -------
CSS_DARK = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
  --bg:#0b1220;
  --panel: rgba(255,255,255,0.035);
  --muted:#9fb0c6;
  --text:#eaf6ff;
  --accent1:#7c3aed;
  --accent2:#06b6d4;
  --accent3:#ff5c8a;
  --glass: rgba(255,255,255,0.05);
  --border: rgba(255,255,255,0.06);
}

/* Sticker heading variables */
:root {
  --sticker-bg: linear-gradient(90deg, var(--accent2), var(--accent3));
  --sticker-color: #041;
}

/* base page */
html, body {
  background:
    radial-gradient(circle at 10% 10%, rgba(124,58,237,0.07), transparent 10%),
    radial-gradient(circle at 90% 90%, rgba(6,182,212,0.06), transparent 10%),
    linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.0)),
    var(--bg) !important;
  color:var(--text) !important;
  font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial;
  margin:0; padding:0;
}

/* tighten streamlit chrome */
section[data-testid="stSidebar"] { background: rgba(255,255,255,0.02) !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stSidebarContent"] { padding-top: 18px; }
div.block-container { padding-top: 16px !important; }

/* anchors: smoother jump + spacing */
[id] { scroll-margin-top: 90px; }

/* wrap */
.wrap { max-width:1200px; margin:22px auto; padding:18px; }

/* Hero */
.hero { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }
.brand { display:flex; gap:14px; align-items:center; }
.logo {
  width:74px; height:74px; border-radius:18px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,var(--accent1),var(--accent2)); font-weight:900; color:#041; font-size:20px;
  box-shadow: 0 18px 50px rgba(2,6,23,0.55); transform: rotate(-8deg);
  animation: logo-tilt 3s infinite ease-in-out;
}
@keyframes logo-tilt { 0% { transform: rotate(-8deg) translateY(0); } 50% { transform: rotate(-2deg) translateY(-6px); } 100% { transform: rotate(-8deg) translateY(0); } }
.title { font-size:32px; font-weight:900; margin:0; letter-spacing:-0.6px; }
.subtitle { color:var(--muted); margin-top:6px; font-size:13px; }

.right-actions { min-width:320px; border-radius:12px; padding:6px; text-align:right; }
.cta {
  display:inline-block; padding:10px 14px; border-radius:12px; font-weight:900; text-decoration:none; cursor:pointer;
  background: linear-gradient(90deg,var(--accent2),var(--accent3)); color:#041;
  box-shadow:0 14px 34px rgba(6,182,212,0.10);
}

/* info strip */
.kpi {
  display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 16px 0;
}
.kpi .pill {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
}
.kpi .pill b { color: var(--text); }

/* grid/cards */
.section { margin-top:22px; }
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:12px; }
.card {
  background: var(--panel);
  border-radius:16px;
  padding:16px;
  position:relative;
  overflow:hidden;
  border: 1px solid var(--border);
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  box-shadow: 0 14px 34px rgba(2,6,23,0.16);
}
.card:hover {
  transform: translateY(-10px) scale(1.01);
  border-color: rgba(6,182,212,0.22);
  box-shadow: 0 42px 120px rgba(2,6,23,0.28);
}

.icon-bubble {
  width:56px; height:56px; border-radius:14px;
  display:flex; align-items:center; justify-content:center;
  font-size:22px; font-weight:900;
  background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  box-shadow: 0 14px 34px rgba(2,6,23,0.20);
  animation: float 3s infinite ease-in-out;
}
@keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-7px);} 100% { transform: translateY(0);} }

.card-title { font-size:15px; font-weight:900; margin-top:10px; }
.card-desc { color:var(--muted); margin-top:8px; font-size:13px; line-height:1.5; }

.ribbon { height:6px; border-radius:10px; margin-top:12px; background: linear-gradient(90deg,var(--accent1),var(--accent2),var(--accent3)); background-size:200% 100%; animation: slide 6s linear infinite; }
@keyframes slide { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

.shimmer {
  position:absolute; inset:0; pointer-events:none;
  background: linear-gradient(90deg, rgba(255,255,255,0.00), rgba(255,255,255,0.05) 50%, rgba(255,255,255,0.00));
  transform:translateX(-140%); animation: shimmer 3.8s linear infinite;
}
@keyframes shimmer { to { transform: translateX(140%); } }

/* Sticker heading styles */
.section-heading { display:flex; align-items:center; gap:14px; margin-top:26px; margin-bottom:8px; }
.section-sticker {
  display:inline-block; padding:6px 14px; font-size:11px; font-weight:900; letter-spacing:1.4px;
  border-radius:8px; text-transform:uppercase; background: var(--sticker-bg); color: var(--sticker-color);
  box-shadow: 0 10px 26px rgba(6,182,212,0.14);
}
.section-title { margin:0; font-size:22px; font-weight:900; letter-spacing:-0.3px; }

/* description */
.desc-box {
  margin-top: 6px;
  margin-bottom: 12px;
  max-width: 860px;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.6;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 14px;
}
.desc-box ul { padding-left: 18px; margin: 6px 0; }

/* prediction panel styling */
.pred-panel {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 14px 34px rgba(2,6,23,0.12);
}
.pred-hint { color: var(--muted); font-size: 13px; line-height: 1.6; }

/* Flowchart styles */
.flowwrap { margin-top:26px; display:flex; align-items:center; justify-content:center; }
.flow { display:flex; align-items:center; gap:18px; flex-wrap:wrap; justify-content:center; }
.flow .step {
  background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  color:var(--text);
  padding:12px 18px;
  border-radius:14px;
  font-weight:800;
  min-width:170px;
  text-align:center;
  border:1px solid var(--border);
  box-shadow: 0 10px 30px rgba(2,6,23,0.12);
  transition: transform .22s ease, box-shadow .22s ease;
}
.flow .step:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 34px 110px rgba(2,6,23,0.28); }
.flow .arrow { width:46px; height:2px; background: linear-gradient(90deg,var(--accent2),var(--accent3)); position:relative; display:inline-block; }
.flow .arrow:after {
  content: ''; position:absolute; right:-6px; top:-6px;
  border-width:8px; border-style:solid;
  border-color: transparent transparent transparent var(--accent3);
  filter: drop-shadow(0 4px 8px rgba(2,6,23,0.12));
}
.flow .arrow.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0% { transform: translateX(0) scaleX(1); } 50% { transform: translateX(3px) scaleX(1.05);} 100% { transform: translateX(0) scaleX(1); } }

/* responsive */
@media (max-width:900px) {
  .grid { grid-template-columns: repeat(2, 1fr); }
  .flow { gap:12px; }
  .flow .step { min-width:150px; padding:10px 12px; font-size:14px; }
  .flow .arrow { width:34px; }
}
@media (max-width:650px) {
  .grid { grid-template-columns: 1fr; }
}

/* Streamlit widgets polish */
div.stButton > button {
  border-radius: 12px !important;
  padding: 10px 14px !important;
  font-weight: 900 !important;
}
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 16px !important;
  border: 1px dashed rgba(255,255,255,0.18) !important;
  background: rgba(255,255,255,0.02) !important;
}
div[data-testid="stRadio"] label { font-weight: 700 !important; }
div[data-testid="stDataFrame"] { border-radius: 16px; overflow:hidden; border: 1px solid var(--border); }
</style>
"""

CSS_LIGHT = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
  --bg:#f7fbff;
  --panel:#ffffff;
  --muted:#475569;
  --text:#071028;
  --accent1:#5b21b6;
  --accent2:#06b6d4;
  --accent3:#ff4d6d;
  --glass: rgba(0,0,0,0.06);
  --border: rgba(2,6,23,0.08);
}

/* Sticker heading variables */
:root {
  --sticker-bg: linear-gradient(90deg, var(--accent2), var(--accent3));
  --sticker-color: #fff;
}

html, body { background: var(--bg) !important; color:var(--text) !important; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; margin:0; padding:0; }
section[data-testid="stSidebar"] { background: rgba(255,255,255,0.8) !important; border-right: 1px solid var(--border) !important; }
div.block-container { padding-top: 16px !important; }
[id] { scroll-margin-top: 90px; }

.wrap { max-width:1200px; margin:22px auto; padding:18px; }

.hero { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }
.brand { display:flex; gap:14px; align-items:center; }
.logo { width:74px; height:74px; border-radius:18px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,var(--accent1),var(--accent2)); font-weight:900; color:#fff; font-size:20px;
  box-shadow: 0 18px 50px rgba(2,6,23,0.06); transform: rotate(-8deg); animation: logo-tilt 3s infinite ease-in-out; }
@keyframes logo-tilt { 0% { transform: rotate(-8deg) translateY(0); } 50% { transform: rotate(-2deg) translateY(-6px); } 100% { transform: rotate(-8deg) translateY(0); } }
.title { font-size:32px; font-weight:900; margin:0; letter-spacing:-0.6px; }
.subtitle { color:var(--muted); margin-top:6px; font-size:13px }

.right-actions { min-width:320px; border-radius:12px; padding:6px; text-align:right; }
.cta { display:inline-block; padding:10px 14px; border-radius:12px; font-weight:900; text-decoration:none; cursor:pointer;
      background: linear-gradient(90deg,var(--accent2),var(--accent3)); color:#fff; box-shadow:0 14px 34px rgba(6,182,212,0.10); }

.kpi { display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 16px 0; }
.kpi .pill { background: rgba(2,6,23,0.03); border: 1px solid var(--border); padding: 8px 12px; border-radius: 999px; font-size: 12px; color: var(--muted); }
.kpi .pill b { color: var(--text); }

.section { margin-top:22px; }
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:12px; }
.card { background: var(--panel); border-radius:16px; padding:16px; position:relative; overflow:hidden; border: 1px solid var(--border); transition: transform .25s ease, box-shadow .25s ease; box-shadow: 0 14px 34px rgba(2,6,23,0.06); }
.card:hover { transform: translateY(-10px) scale(1.01); box-shadow: 0 36px 90px rgba(2,6,23,0.10); }

.icon-bubble { width:56px; height:56px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:900; background: rgba(2,6,23,0.03); box-shadow: 0 10px 26px rgba(2,6,23,0.06); animation: float 3s infinite ease-in-out; }
@keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-7px);} 100% { transform: translateY(0);} }

.card-title { font-size:15px; font-weight:900; margin-top:10px; }
.card-desc { color:var(--muted); margin-top:8px; font-size:13px; line-height:1.5; }

.ribbon { height:6px; border-radius:10px; margin-top:12px; background: linear-gradient(90deg,var(--accent1),var(--accent2),var(--accent3)); background-size:200% 100%; animation: slide 6s linear infinite; }
@keyframes slide { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

.shimmer { position:absolute; inset:0; pointer-events:none; background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(2,6,23,0.05) 50%, rgba(255,255,255,0.0)); transform:translateX(-140%); animation: shimmer 3.8s linear infinite; }
@keyframes shimmer { to { transform: translateX(140%); } }

.section-heading { display:flex; align-items:center; gap:14px; margin-top:26px; margin-bottom:8px; }
.section-sticker { display:inline-block; padding:6px 14px; font-size:11px; font-weight:900; letter-spacing:1.4px; border-radius:8px; text-transform:uppercase; background: var(--sticker-bg); color: var(--sticker-color); box-shadow: 0 10px 26px rgba(6,182,212,0.10); }
.section-title { margin:0; font-size:22px; font-weight:900; letter-spacing:-0.3px; }

.desc-box { margin-top: 6px; margin-bottom: 12px; max-width: 860px; font-size: 13px; color: var(--muted); line-height: 1.6; background: rgba(255,255,255,0.7); border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }
.desc-box ul { padding-left: 18px; margin: 6px 0; }

.pred-panel { background: rgba(255,255,255,0.75); border: 1px solid var(--border); border-radius: 18px; padding: 16px; box-shadow: 0 14px 34px rgba(2,6,23,0.05); }
.pred-hint { color: var(--muted); font-size: 13px; line-height: 1.6; }

.flowwrap { margin-top:26px; display:flex; align-items:center; justify-content:center; }
.flow { display:flex; align-items:center; gap:18px; flex-wrap:wrap; justify-content:center; }
.flow .step { background: rgba(255,255,255,0.95); color:var(--text); padding:12px 18px; border-radius:14px; font-weight:900; min-width:170px; text-align:center; border:1px solid var(--border); box-shadow: 0 10px 30px rgba(2,6,23,0.06); transition: transform .22s ease, box-shadow .22s ease; }
.flow .step:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 30px 90px rgba(2,6,23,0.12); }
.flow .arrow { width:46px; height:2px; background: linear-gradient(90deg,var(--accent2),var(--accent3)); position:relative; display:inline-block; }
.flow .arrow:after { content: ''; position:absolute; right:-6px; top:-6px; border-width:8px; border-style:solid; border-color: transparent transparent transparent var(--accent3); }
.flow .arrow.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0% { transform: translateX(0) scaleX(1); } 50% { transform: translateX(3px) scaleX(1.05);} 100% { transform: translateX(0) scaleX(1); } }

@media (max-width:900px) { .grid { grid-template-columns: repeat(2, 1fr); } .flow { gap:12px; } .flow .step { min-width:150px; padding:10px 12px; font-size:14px; } .flow .arrow { width:34px; } }
@media (max-width:650px) { .grid { grid-template-columns: 1fr; } }

div.stButton > button { border-radius: 12px !important; padding: 10px 14px !important; font-weight: 900 !important; }
div[data-testid="stFileUploaderDropzone"] { border-radius: 16px !important; border: 1px dashed rgba(2,6,23,0.20) !important; background: rgba(255,255,255,0.7) !important; }
div[data-testid="stDataFrame"] { border-radius: 16px; overflow:hidden; border: 1px solid var(--border); }
</style>
"""

# apply chosen CSS
st.markdown(CSS_DARK if st.session_state.theme == "dark" else CSS_LIGHT, unsafe_allow_html=True)

# ------ Page content -------
st.markdown('<div class="wrap">', unsafe_allow_html=True)

# HERO
st.markdown(
    """
<div class="hero">
  <div class="brand">
    <div class="logo">TF</div>
    <div>
      <div class="title">TraceFinder</div>
      <div class="subtitle">Scanner & Camera Source Identification — PRNU + Metadata</div>
    </div>
  </div>
  <div class="right-actions">
    <div style="font-size:13px; color:var(--muted); margin-bottom:6px;">Project brief</div>
    <div class="cta">Landing • Features • Use-cases</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# KPIs (UI-only enhancement)
st.markdown(
    """
<div class="kpi">
  <div class="pill"><b>Models:</b> RF • SVM • CNN • Hybrid</div>
  <div class="pill"><b>Signals:</b> PRNU residual + metadata</div>
  <div class="pill"><b>Use:</b> Forensics • KYC • Research</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---- 6-point project description (just below title) ----
desc_html = """
<div class="desc-box">
  <ul>
    <li>Identify the scanner brand/model used to produce a scanned document by analysing device-specific noise patterns.</li>
    <li>Support digital forensics, document authentication and legal evidence verification workflows.</li>
    <li>Extract scanner-specific features such as noise residuals, entropy, edge density and other metadata signals.</li>
    <li>Train machine learning models (Random Forest &amp; SVM) to distinguish between multiple scanner devices.</li>
    <li>Evaluate performance using accuracy, precision, recall, F1-score and confusion matrix visualisations.</li>
    <li>Provide a simple UI where users upload a scan and receive the predicted scanner model with confidence.</li>
  </ul>
</div>
"""
st.markdown(desc_html, unsafe_allow_html=True)

# PDF viewer + button
st.markdown('<div style="display:flex; gap:14px; align-items:center; margin-bottom:12px;">', unsafe_allow_html=True)
if default_pdf_bytes:
    pdf_uri = pdf_to_data_uri(default_pdf_bytes[:MAX_EMBED_BYTES])
    st.markdown(f'<a class="cta" href="{pdf_uri}" target="_blank">Open Project Brief (PDF)</a>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color:#f3c; font-weight:800;">Project PDF not found in project folder.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# FEATURES
st.markdown('<div class="section" id="features">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">FEATURES</div>
    <h3 class="section-title">Core Capabilities</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

features = [
    ("🔬", "Noise Residual Extraction", "Wavelet denoising + residual subtraction to reveal sensor PRNU patterns."),
    ("📊", "Compact Metadata Set", "Entropy, skewness, edge density and file-size metrics for explainable signals."),
    ("🧩", "Patch-based Multi-Scale", "Overlapping patches to boost sensor fingerprint and reduce content leakage."),
    ("⚡", "Fast CPU Inference", "Random Forest baseline for quick, low-cost predictions and prototypes."),
]
for icon, title, desc in features:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# USE CASES
st.markdown('<div class="section" id="use-cases">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">USE CASES</div>
    <h3 class="section-title">Real-World Applications</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

uses = [
    ("🛡️", "Digital Forensics", "Attribute scanned evidence to specific scanner models for investigations."),
    ("🏦", "Banking & KYC", "Detect reused or forged scans in identity verification flows."),
    ("🏛️", "Government Services", "Validate IDs and certificates to prevent fraud at scale."),
    ("🔬", "Research", "Benchmark PRNU/device-identification methods and datasets."),
]
for icon, title, desc in uses:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ADVANTAGES
st.markdown('<div class="section" id="advantages">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">ADVANTAGES</div>
    <h3 class="section-title">Why TraceFinder Stands Out</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

advantages = [
    ("🏆", "High baseline accuracy", "Strong per-scanner performance using residual + metadata features."),
    ("🔍", "Explainable", "Feature-level evidence supports forensic reporting and traceability."),
    ("🔐", "Privacy-first", "Offline processing suitable for sensitive documents and workflows."),
    ("🔧", "Extensible", "Easily swap classifiers or add scanner classes as datasets grow."),
]
for icon, title, desc in advantages:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION DEMO SECTION ----------
st.markdown('<div class="section" id="prediction">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">PREDICTION</div>
    <h3 class="section-title">Try TraceFinder on Your Own Scan</h3>
  </div>
""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown(
        """
<div class="pred-panel">
  <div class="pred-hint">
    <b>How prediction works:</b><br><br>
    1. Upload a scanned document image (TIFF / PNG / JPEG).<br>
    2. The app converts it to grayscale and resizes to 512×512.<br>
    3. Scanner-specific metadata features are extracted:
       width, height, aspect ratio, file size (KB), mean/std intensity, skewness, kurtosis, entropy, edge density.<br>
    4. Features are normalized using the same scaler used in training.<br>
    5. A trained model (Random Forest, SVM, CNN, or Hybrid CNN) predicts the scanner class.<br>
    6. RF/SVM/Hybrid can also display per-class probabilities.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown("**Upload a scan and choose a model**")
    uploaded = st.file_uploader(
        "Upload scanned image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="predict_upload",
    )
    model_choice = st.radio(
        "Select model",
        ["Random Forest", "SVM", "CNN", "Hybrid CNN"],
        horizontal=True,
        key="predict_model_choice",
    )

if uploaded is not None:
    # Preview image
    try:
        file_bytes = uploaded.getvalue()
        arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded scan")
    except Exception:
        pass

    if st.button("Run Prediction", type="primary"):

        # Block RF/SVM if models missing
        if model_choice in ["Random Forest", "SVM"] and not SCALER_PATH.exists():
            st.error("RF / SVM models not found. Please select CNN or Hybrid CNN.")
            st.stop()

        try:
            pred_label, probs, class_names = run_prediction(uploaded, model_choice)
            st.success(f"Predicted Scanner: **{pred_label}**")

            if probs is not None and class_names is not None:
                prob_df = (
                    pd.DataFrame(
                        {
                            "Scanner": list(class_names),
                            "Probability": list(probs),
                        }
                    )
                    .sort_values("Probability", ascending=False)
                )
                st.markdown("**Per-class probabilities:**")
                st.dataframe(prob_df, width="content")
            elif model_choice == "CNN":
                st.info("CNN currently shows only the top predicted scanner (probability table hidden).")
            else:
                st.info("This model does not expose probability scores.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a scanned image to run prediction.")

st.markdown("</div>", unsafe_allow_html=True)

# Flowchart
st.markdown('<div class="section" id="flow">', unsafe_allow_html=True)

flow_html = """
<div class="flowwrap">
  <div class="flow" aria-label="Project flowchart">
    <div class="step" title="1. Data Collection">1. Data Collection<br><small>Scan & label</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="2. Preprocessing">2. Preprocessing<br><small>Resize, denoise</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="3. Feature Extraction">3. Feature Extraction<br><small>PRNU, metadata</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="4. Model Training">4. Model Training<br><small>RF / SVM / CNN / Hybrid CNN</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="5. Deploy & UI">5. Deploy & UI<br><small>Streamlit app</small></div>
  </div>
</div>
"""
st.markdown(flow_html, unsafe_allow_html=True)

st.markdown(
    """
<div style="margin-top:18px; text-align:center; color:var(--muted); font-size:12px;">
  Developed by <b>Harsh Pandey</b> • TraceFinder UI • Streamlit Deployment
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# close wrapper div
st.markdown("</div>", unsafe_allow_html=True)