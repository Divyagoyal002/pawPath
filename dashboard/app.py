"""PawPath owner/vet dashboard (Streamlit).

Upload a session CSV (firmware schema: millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps),
or pick a synthetic sample session, and view:
  - raw + preprocessed gait traces
  - a gait classification (healthy/orthopedic/neurological) using the
    trained Random Forest model if available, else a heuristic fallback
  - limb-symmetry indices (FR-5)
  - a longitudinal per-dog trend log (FR-6) stored in dashboard/session_log.csv
  - a downloadable risk report for vet review (FR-8)

Run with: streamlit run dashboard/app.py
"""

import glob
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_processing.filters import preprocess_imu
from signal_processing.windowing import window_signal
from signal_processing.features import extract_feature_matrix

DASHBOARD_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DASHBOARD_DIR, "..", "data", "synthetic")
MODEL_PATH = os.path.join(DASHBOARD_DIR, "..", "ml", "classical", "rf_model.joblib")
SESSION_LOG_PATH = os.path.join(DASHBOARD_DIR, "session_log.csv")

st.set_page_config(page_title="PawPath Gait Dashboard", layout="wide")


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def list_synthetic_samples():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.csv")))


def heuristic_classify(feature_row: dict) -> str:
    """Fallback classifier (no trained model yet) using two literature-backed
    signals: high-frequency energy ratio (motor-control smoothness) and
    gx/gy correlation (limb-loading symmetry)."""
    high_freq = np.mean([v for k, v in feature_row.items() if k.endswith("band_high_ratio")])
    correlation = feature_row.get("gx_gy_correlation", 1.0)

    if high_freq > 0.25:
        return "neurological"
    if correlation < 0.4:
        return "orthopedic"
    return "healthy"


def compute_symmetry_index(preprocessed: np.ndarray) -> float:
    """FR-5: crude left/right loading symmetry from gyro-Y stride amplitude
    alternation. 1.0 = perfectly symmetric strides, lower = more asymmetric."""
    gy = preprocessed[:, 4]
    peaks = gy[1:-1][(gy[1:-1] > gy[:-2]) & (gy[1:-1] > gy[2:]) & (gy[1:-1] > 0)]
    if len(peaks) < 4:
        return 1.0
    even_amp = np.mean(peaks[0::2])
    odd_amp = np.mean(peaks[1::2])
    if even_amp + odd_amp == 0:
        return 1.0
    return float(1.0 - abs(even_amp - odd_amp) / (even_amp + odd_amp))


def log_session_result(dog_id: str, label: str, symmetry: float):
    row = pd.DataFrame(
        [{"timestamp": datetime.now().isoformat(timespec="seconds"), "dog_id": dog_id, "prediction": label, "symmetry_index": symmetry}]
    )
    if os.path.exists(SESSION_LOG_PATH):
        row.to_csv(SESSION_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(SESSION_LOG_PATH, index=False)


st.title("🐾 PawPath — Canine Gait Dashboard")
st.caption("Non-invasive IMU-based screening for early spinal injury, lameness and arthritis indicators.")

with st.sidebar:
    st.header("Session input")
    dog_id = st.text_input("Dog ID", value="demo_dog_01")
    source = st.radio("Data source", ["Upload CSV", "Synthetic sample"])

    session_df = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload IMU session CSV", type="csv")
        if uploaded is not None:
            session_df = pd.read_csv(uploaded)
    else:
        samples = list_synthetic_samples()
        if not samples:
            st.warning("No synthetic samples found. Run data/synthetic_gait_generator.py first.")
        else:
            choice = st.selectbox("Sample session", samples, format_func=os.path.basename)
            session_df = pd.read_csv(choice)

if session_df is None:
    st.info("Upload a session CSV or pick a synthetic sample from the sidebar to see a gait report.")
    st.stop()

raw = session_df[["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]].to_numpy()
preprocessed = preprocess_imu(raw)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw accelerometer (g)")
    st.line_chart(pd.DataFrame(raw[:, :3], columns=["ax", "ay", "az"]))
with col2:
    st.subheader("Preprocessed (gravity-removed, denoised)")
    st.line_chart(pd.DataFrame(preprocessed[:, :3], columns=["ax", "ay", "az"]))

windows = window_signal(preprocessed)
model_bundle = load_model()

if windows.shape[0] == 0:
    st.warning("Session too short to window (need at least 120 samples / ~1.2s at 100Hz).")
    st.stop()

feature_matrix, feature_names = extract_feature_matrix(windows)

if model_bundle is not None:
    pipeline = model_bundle["pipeline"]
    window_predictions = pipeline.predict(feature_matrix)
    values, counts = np.unique(window_predictions, return_counts=True)
    overall_label = values[np.argmax(counts)]
    st.caption("Classification via trained Random Forest model.")
else:
    per_window_labels = [heuristic_classify(dict(zip(feature_names, row))) for row in feature_matrix]
    values, counts = np.unique(per_window_labels, return_counts=True)
    overall_label = values[np.argmax(counts)]
    st.caption("⚠️ No trained model found — using heuristic fallback. Train one with ml/classical/train_rf.py.")

symmetry = compute_symmetry_index(preprocessed)

st.subheader("Gait Report")
report_col1, report_col2, report_col3 = st.columns(3)
report_col1.metric("Classification", overall_label.capitalize())
report_col2.metric("Symmetry index", f"{symmetry:.2f}", help="1.0 = perfectly symmetric strides")
report_col3.metric("Windows analyzed", windows.shape[0])

if overall_label != "healthy":
    st.error(f"⚠️ Abnormal gait pattern detected ({overall_label}). Recommend veterinary follow-up.")
else:
    st.success("No abnormal gait pattern detected in this session.")

log_session_result(dog_id, overall_label, symmetry)

st.subheader(f"Longitudinal trend for {dog_id}")
if os.path.exists(SESSION_LOG_PATH):
    log_df = pd.read_csv(SESSION_LOG_PATH)
    dog_log = log_df[log_df["dog_id"] == dog_id]
    if len(dog_log) > 1:
        st.line_chart(dog_log.set_index("timestamp")["symmetry_index"])
    st.dataframe(dog_log.tail(20), use_container_width=True)

st.download_button(
    "Download vet report (CSV)",
    data=pd.DataFrame(
        [{"dog_id": dog_id, "timestamp": datetime.now().isoformat(timespec="seconds"), "classification": overall_label, "symmetry_index": symmetry}]
    ).to_csv(index=False),
    file_name=f"pawpath_report_{dog_id}.csv",
    mime="text/csv",
)
