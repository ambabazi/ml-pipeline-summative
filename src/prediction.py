"""Load the live model and predict one drawing."""
import joblib
import pandas as pd

from src.model import MODELS, REGISTRY, load_registry
from src.preprocessing import FEATURES, extract_features

_cache = {"mtime": None, "model": None, "version": None}


def get_model():
    """Return the active model; reload only if the registry changed (e.g. another container retrained)."""
    mtime = REGISTRY.stat().st_mtime if REGISTRY.exists() else None
    if mtime is None:
        raise RuntimeError("No trained model yet - run `python -m src.model` first")
    if mtime != _cache["mtime"]:
        reg = load_registry()
        entry = next(v for v in reg["versions"] if v["version"] == reg["active"])
        _cache.update(mtime=mtime, model=joblib.load(MODELS / entry["file"]), version=entry["version"])
    return _cache["model"], _cache["version"]


def predict(source, threshold=0.5):
    """Image path or bytes -> prediction, probability, and the features behind it."""
    model, version = get_model()
    feats = extract_features(source, check_input=True)
    prob = float(model.predict_proba(pd.DataFrame([feats])[FEATURES])[0, 1])
    label = "parkinson" if prob >= threshold else "healthy"
    return {"prediction": label,
            "probability_parkinson": round(prob, 4),
            "confidence": round(max(prob, 1 - prob), 4),
            "model_version": version,
            "features": {k: round(v, 4) for k, v in feats.items()}}