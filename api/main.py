"""NeuroTrace REST API: predict, upload, retrain, monitor."""
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from src.model import DATA, LABELS, load_registry, run_training
from src.prediction import get_model, predict

app = FastAPI(title="NeuroTrace API", version="1.0",
              description="Parkinson's drawing-pattern screening prototype - NOT a diagnostic tool.")

STARTED = time.time()                                    # for uptime
CONTAINER = os.getenv("HOSTNAME", "local")               # Docker sets HOSTNAME = container id -> shows load balancing
stats = {"requests": 0, "errors": 0, "predictions": 0, "total_latency_ms": 0.0}   # simple in-memory counters
retrain_state = {"status": "idle", "started_at": None, "finished_at": None, "result": None, "error": None}
retrain_lock = threading.Lock()                          # only one retraining at a time
ALLOWED = {".jpg", ".jpeg", ".png"}                      # accepted image types


@app.middleware("http")
async def track_latency(request: Request, call_next):
    """Runs around every request: count it and time it."""
    start = time.perf_counter()                          # precise timer start
    response = await call_next(request)                  # run the actual endpoint
    ms = (time.perf_counter() - start) * 1000            # elapsed time in milliseconds
    stats["requests"] += 1
    stats["total_latency_ms"] += ms
    if response.status_code >= 400:                      # 4xx/5xx = error
        stats["errors"] += 1
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"   # latency visible to clients / Locust
    response.headers["X-Container"] = CONTAINER          # which container answered
    return response


@app.get("/health")
def health():
    """Uptime + live model version (used by the dashboard and Docker health checks)."""
    try:
        _, version = get_model()
        model_status = "loaded"
    except Exception:                                    # no model trained yet
        version, model_status = None, "missing"
    return {"status": "ok", "model_status": model_status, "model_version": version,
            "uptime_seconds": round(time.time() - STARTED, 1), "container": CONTAINER,
            "started_at": datetime.fromtimestamp(STARTED, timezone.utc).isoformat()}


@app.get("/stats")
def get_stats():
    """Request counters and average latency for this container."""
    n = stats["requests"] or 1                           # avoid dividing by zero
    return {**stats, "avg_latency_ms": round(stats["total_latency_ms"] / n, 2), "container": CONTAINER}


@app.post("/predict")
async def predict_drawing(file: UploadFile = File(...)):
    """Upload ONE spiral/meander image -> prediction + confidence + the features behind it."""
    if Path(file.filename or "").suffix.lower() not in ALLOWED:
        raise HTTPException(400, "Please upload a .jpg or .png image")   # reject non-images early
    try:
        result = predict(await file.read())              # read the bytes and predict
    except ValueError as e:                              # unreadable image
        raise HTTPException(400, str(e))
    stats["predictions"] += 1
    return {**result, "disclaimer": "Research prototype - not a medical diagnosis."}


@app.post("/upload")
async def upload_training_data(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload LABELLED images; they are added to the training pool for the next retrain."""
    if label not in LABELS:
        raise HTTPException(400, f"label must be one of {LABELS}")
    folder = DATA / "uploads" / label                    # data/uploads/healthy or data/uploads/parkinson
    folder.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = Path(f.filename or "").name               # strip any folder path from the filename (security)
        if Path(name).suffix.lower() not in ALLOWED:
            rejected.append(name)
            continue
        dest = folder / name
        if dest.exists():                                # don't overwrite: add a short unique suffix
            dest = folder / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
        dest.write_bytes(await f.read())                 # save the image to disk
        saved.append(dest.name)
    return {"label": label, "saved": len(saved), "rejected": rejected,
            "total_uploaded": {l: len(list((DATA / "uploads" / l).glob("*"))) for l in LABELS}}


def _retrain_job():
    """Runs in a background thread so the API stays responsive while training."""
    try:
        retrain_state["result"] = run_training()
        retrain_state["status"] = "completed"
    except Exception as e:
        retrain_state.update(status="failed", error=str(e))
    finally:
        retrain_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        retrain_lock.release()                           # allow the next retrain


@app.post("/retrain")
def trigger_retrain():
    """The retrain button: start training in the background and return immediately."""
    if not retrain_lock.acquire(blocking=False):         # someone already pressed it
        raise HTTPException(409, "Retraining already running")
    retrain_state.update(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                         finished_at=None, result=None, error=None)
    threading.Thread(target=_retrain_job, daemon=True).start()
    return {"message": "Retraining started", "check": "/retrain/status"}


@app.get("/retrain/status")
def retrain_status():
    return retrain_state                                 # running / completed / failed + new version's metrics


@app.get("/models")
def models():
    """Every model version with its test metrics - v1 vs v2 comparison for the dashboard."""
    return load_registry()