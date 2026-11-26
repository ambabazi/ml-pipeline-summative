"""Load the live model and predict one drawing."""
import joblib
import pandas as pd

from src.model import MODELS, REGISTRY, load_registry
from src.preprocessing import FEATURES, extract_features

_cache = {"mtime": None, "model": None, "version": None}  # keep the model in memory between requests


def get_model():
    """Return the active model; reload only if the registry changed (e.g. another container retrained)."""
    mtime = REGISTRY.stat().st_mtime if REGISTRY.exists() else None   # registry's "last modified" time
    if mtime is None:
        raise RuntimeError("No trained model yet - run `python -m src.model` first")
    if mtime != _cache["mtime"]:                               # registry file changed -> new active version?
        reg = load_registry()
        entry = next(v for v in reg["versions"] if v["version"] == reg["active"])  # find the live version
        _cache.update(mtime=mtime, model=joblib.load(MODELS / entry["file"]), version=entry["version"])
    return _cache["model"], _cache["version"]


def predict(source, threshold=0.5):
    """Image path or bytes -> prediction, probability, and the features behind it."""
    model, version = get_model()
    feats = extract_features(source, check_input=True)                                        # same 9 features as training
    prob = float(model.predict_proba(pd.DataFrame([feats])[FEATURES])[0, 1])  # P(Parkinson's pattern)
    label = "parkinson" if prob >= threshold else "healthy"
    return {"prediction": label,
            "probability_parkinson": round(prob, 4),
            "confidence": round(max(prob, 1 - prob), 4),       # certainty in whichever class was chosen
            "model_version": version,                          # which version answered (v1 or v2)
            "features": {k: round(v, 4) for k, v in feats.items()}}   # returned so the UI can explain the prediction