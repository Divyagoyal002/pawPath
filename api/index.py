"""PawPath gait dashboard — FastAPI + server-rendered HTML.

Replaces the earlier Streamlit prototype (dashboard/app.py) so the app is
small enough to deploy on Vercel: only FastAPI + scikit-learn + the
trained Random Forest classifier are required at request time. The
PyTorch/1D-CNN track (ml/deep_learning/) stays local-only — it is far
over Vercel's deployment size limit and is not needed for inference here,
since the deployed classifier is the classical Random Forest model.

Routes:
  GET  /            form: pick a synthetic sample or upload a session CSV
  POST /analyze     run the pipeline, render charts + report
  GET  /report/{id} download that dog's logged session history as CSV
"""

import csv
import glob
import io
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from signal_processing.filters import preprocess_imu  # noqa: E402
from signal_processing.features import extract_feature_matrix  # noqa: E402
from signal_processing.windowing import window_signal  # noqa: E402

APP_DIR = os.path.dirname(__file__)
# A handful of demo sessions bundled with the app itself (see api/samples/)
# so the deployed page has something to try without requiring an upload.
# The full synthetic training set (data/synthetic/) is gitignored and not
# deployed — regenerate it locally with data/synthetic_gait_generator.py.
DATA_DIR = os.path.join(APP_DIR, "samples")
MODEL_PATH = os.path.join(ROOT_DIR, "ml", "classical", "rf_model.joblib")

# Vercel's filesystem is read-only outside /tmp, and /tmp is not shared
# across invocations/instances — session history only really persists
# when this app runs somewhere with a persistent disk (e.g. a VM, or
# `uvicorn` locally). See README for the production-persistence caveat.
SESSION_LOG_PATH = os.path.join(os.environ.get("PAWPATH_LOG_DIR", APP_DIR), "session_log.csv")

app = FastAPI(title="PawPath")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

_model_bundle = None
_model_load_attempted = False


def get_model():
    global _model_bundle, _model_load_attempted
    if not _model_load_attempted:
        _model_load_attempted = True
        if os.path.exists(MODEL_PATH):
            _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def list_synthetic_samples():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.csv")))


def heuristic_classify(feature_row: dict) -> str:
    """Fallback used only if rf_model.joblib is missing (see ml/classical/train_rf.py)."""
    high_freq_keys = [v for k, v in feature_row.items() if k.endswith("band_high_ratio")]
    high_freq = float(np.mean(high_freq_keys)) if high_freq_keys else 0.0
    correlation = feature_row.get("gx_gy_correlation", 1.0)
    if high_freq > 0.25:
        return "neurological"
    if correlation < 0.4:
        return "orthopedic"
    return "healthy"


def compute_symmetry_index(preprocessed: np.ndarray) -> float:
    """FR-5: crude stride-to-stride loading symmetry from gyro-Y peak amplitude alternation."""
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
    is_new = not os.path.exists(SESSION_LOG_PATH)
    try:
        with open(SESSION_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "dog_id", "prediction", "symmetry_index"])
            writer.writerow(
                [datetime.now(timezone.utc).isoformat(timespec="seconds"), dog_id, label, f"{symmetry:.4f}"]
            )
    except OSError:
        pass  # read-only filesystem (e.g. Vercel) — trend log simply won't persist


def read_dog_log(dog_id: str):
    if not os.path.exists(SESSION_LOG_PATH):
        return []
    with open(SESSION_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["dog_id"] == dog_id]


def analyze_csv_bytes(raw_bytes: bytes):
    data = np.loadtxt(io.StringIO(raw_bytes.decode("utf-8")), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    raw = data[:, 1:7]  # drop millis column
    preprocessed = preprocess_imu(raw)
    windows = window_signal(preprocessed)
    return raw, preprocessed, windows


def downsample(arr, max_points=300):
    step = max(1, len(arr) // max_points)
    return [round(float(v), 4) for v in arr[::step]]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"samples": list_synthetic_samples(), "result": None, "error": None},
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    dog_id: str = Form(...),
    sample: str = Form(None),
    file: UploadFile = File(None),
):
    samples = list_synthetic_samples()

    if file is not None and file.filename:
        raw_bytes = await file.read()
    elif sample:
        with open(os.path.join(DATA_DIR, sample), "rb") as f:
            raw_bytes = f.read()
    else:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"samples": samples, "result": None, "error": "Upload a CSV or choose a sample session."},
        )

    try:
        raw, preprocessed, windows = analyze_csv_bytes(raw_bytes)
    except (ValueError, IndexError):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"samples": samples, "result": None, "error": "Could not parse that CSV — expected the firmware schema: millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps."},
        )

    if windows.shape[0] == 0:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"samples": samples, "result": None, "error": "Session too short to analyze (need at least 120 samples, ~1.2s at 100Hz)."},
        )

    feature_matrix, feature_names = extract_feature_matrix(windows)
    model_bundle = get_model()
    used_model = model_bundle is not None
    if used_model:
        predictions = model_bundle["pipeline"].predict(feature_matrix)
    else:
        predictions = [heuristic_classify(dict(zip(feature_names, row))) for row in feature_matrix]

    values, counts = np.unique(predictions, return_counts=True)
    overall_label = str(values[np.argmax(counts)])
    symmetry = compute_symmetry_index(preprocessed)

    log_session_result(dog_id, overall_label, symmetry)
    dog_log = read_dog_log(dog_id)

    chart_data = {
        "raw": {axis: downsample(raw[:, i]) for i, axis in enumerate(["ax", "ay", "az"])},
        "preprocessed": {axis: downsample(preprocessed[:, i]) for i, axis in enumerate(["ax", "ay", "az"])},
        "trend_values": [float(r["symmetry_index"]) for r in dog_log],
        "trend_labels": [r["timestamp"][:19] for r in dog_log],
    }

    result = {
        "dog_id": dog_id,
        "classification": overall_label,
        "symmetry": round(symmetry, 3),
        "windows_analyzed": int(windows.shape[0]),
        "used_model": used_model,
        "is_abnormal": overall_label != "healthy",
        "chart_data": chart_data,
        "log_rows": list(reversed(dog_log[-20:])),
    }
    return templates.TemplateResponse(
        request, "index.html", {"samples": samples, "result": result, "error": None}
    )


@app.get("/report/{dog_id}", response_class=PlainTextResponse)
def report(dog_id: str):
    dog_log = read_dog_log(dog_id)
    if not dog_log:
        return PlainTextResponse("No sessions logged for this dog yet.", status_code=404)
    lines = ["timestamp,dog_id,prediction,symmetry_index"]
    lines += [f"{r['timestamp']},{r['dog_id']},{r['prediction']},{r['symmetry_index']}" for r in dog_log]
    return PlainTextResponse(
        "\n".join(lines),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=pawpath_report_{dog_id}.csv"},
    )
