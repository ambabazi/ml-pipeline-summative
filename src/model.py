"""Train, evaluate, version and retrain the NeuroTrace Random Forest."""
import hashlib
import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing import FEATURES, extract_features
from src.uploads import read_uploads

ROOT = Path(__file__).resolve().parent.parent       # project root (src/..), works from any working directory
DATA = ROOT / "data"                                  # data/train, data/test, data/uploads
MODELS = ROOT / "models"                              # saved .pkl files + registry
REGISTRY = MODELS / "registry.json"                   # list of all model versions + which one is live
LABELS = ["healthy", "parkinson"]                     # class 0, class 1


def person_of(path):
    """'healthy/spiral_0068-1.jpg' -> 'healthy_0068' so one person's drawings stay together."""
    path = Path(path)
    return f"{path.parent.name}_{path.stem.split('_')[-1].split('-')[0]}"   # label + person number


def image_table(paths):
    """List of image paths -> DataFrame with label, person and the 9 features."""
    rows = []
    for p in paths:
        row = extract_features(p)                     # the 9 numbers
        row.update(path=str(p), label=Path(p).parent.name, person=person_of(p))  # folder name = label
        rows.append(row)
    df = pd.DataFrame(rows)
    df["y"] = (df.label == "parkinson").astype(int)   # 1 = Parkinson's, 0 = healthy
    return df


def training_images():
    """The original fitting pool, excluding the held-back upload batch."""
    roles = pd.read_csv(DATA / "train_roles.csv")
    return [DATA / "train" / filename for filename in roles[roles.role != "new_batch"].file]


@lru_cache(maxsize=2)
def fixed_image_digests(data):
    """Index the bundled base and evaluation images once per process."""
    roles = pd.read_csv(data / "train_roles.csv")
    paths = [data / "train" / filename for filename in roles[roles.role != "new_batch"].file]
    paths.extend((data / "test").glob("*/*.jpg"))
    return {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def active_entry(registry):
    return next((v for v in registry["versions"] if v["version"] == registry["active"]), None)


def used_upload_digests(current):
    if current is None:
        return set()
    if "used_upload_digests" in current:
        return set(current["used_upload_digests"])
    if not current["n_uploaded_images"] and not current.get("parent_version"):
        return set()
    raise ValueError("The active model has no upload history. Deploy a model with a verified training history.")


def upload_batch(current, *, include_content=False):
    used = used_upload_digests(current)
    rows = read_uploads(DATA / "uploads", include_content=include_content)
    stored = {row[0]: row for row in rows}
    # Older deployments may have saved uploads as files before SQLite was introduced.
    for label in LABELS:
        for path in sorted((DATA / "uploads" / label).glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            row = (digest, label, path.name)
            stored.setdefault(digest, (*row, content) if include_content else row)
    excluded = used | fixed_image_digests(DATA) if stored else used
    pending = [row for digest, row in stored.items() if digest not in excluded]
    counts = {"stored": len(stored), "used": len(used & stored.keys()), "pending": len(pending)}
    return counts, pending


def upload_status():
    return upload_batch(active_entry(load_registry()))[0]


def train(df):
    """Tune and fit a Random Forest with person-grouped cross-validation (same as the notebook)."""
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid={"n_estimators": [100, 300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
        cv=StratifiedGroupKFold(n_splits=5), scoring="roc_auc", n_jobs=-1)   # n_jobs=-1: use all CPU cores
    grid.fit(df[FEATURES], df.y, groups=df.person)    # groups = no person in two folds
    return grid.best_estimator_, grid.best_params_


def evaluate(model, test_df):
    """All the metrics, on the untouched test set."""
    prob = model.predict_proba(test_df[FEATURES])[:, 1]   # P(Parkinson's) for each test image
    pred = (prob >= 0.5).astype(int)                      # decision at 0.5
    y = test_df.y
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "specificity": round(recall_score(y, pred, pos_label=0), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
    }


def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())       # existing history of versions
    return {"active": None, "versions": []}           # first run: empty registry


def save_registry(reg):
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    tmp.replace(REGISTRY)                             # atomic swap: other containers never read a half-written file


def run_training(min_auc_drop=0.02):
    """Full pipeline: gather data -> features -> tune -> evaluate -> save new version -> promote if good enough."""
    reg = load_registry()
    current = active_entry(reg)
    used = used_upload_digests(current)
    _, uploaded_rows = upload_batch(current, include_content=True)
    if current is not None and not uploaded_rows:
        raise ValueError("No new uploads are pending for the active model.")

    MODELS.mkdir(exist_ok=True)
    train_df = image_table(training_images())
    test_df = image_table(sorted((DATA / "test").glob("*/*.jpg")))
    if uploaded_rows:
        features = [
            dict(extract_features(content), y=int(label == "parkinson"),
                 person=person_of(Path(label) / filename))
            for _, label, filename, content in uploaded_rows
        ]
        train_df = pd.concat([train_df, pd.DataFrame(features)], ignore_index=True)
    new_digests = {row[0] for row in uploaded_rows}

    if current is None:
        model, params = train(train_df)
    else:
        # Preserve the active trees and fit a new block on the expanded training pool.
        model = joblib.load(MODELS / current["file"])
        classes = model.classes_
        weights = compute_class_weight("balanced", classes=classes, y=train_df.y)
        model.set_params(warm_start=True, n_estimators=model.n_estimators + 50,
                         class_weight=dict(zip(classes, weights)))
        model.fit(train_df[FEATURES], train_df.y)
        params = {key: model.get_params()[key]
                  for key in ("n_estimators", "max_depth", "min_samples_leaf", "warm_start")}

    metrics = evaluate(model, test_df)
    version = len(reg["versions"]) + 1
    file = f"neurotrace_rf_v{version}.pkl"
    joblib.dump(model, MODELS / file)

    # Quality gate: only go live if ROC-AUC is not worse than the live model by more than min_auc_drop
    promoted = current is None or metrics["roc_auc"] >= current["metrics"]["roc_auc"] - min_auc_drop

    entry = {"version": version, "file": file, "trained_at": datetime.now(timezone.utc).isoformat(),
             "n_train_images": len(train_df), "n_uploaded_images": len(uploaded_rows),
             "used_upload_digests": sorted(used | new_digests),
             "params": params, "metrics": metrics, "promoted": promoted,
             "parent_version": current["version"] if current else None}
    reg["versions"].append(entry)                     # keep a record even if not promoted
    if promoted:
        reg["active"] = version                       # switch the live model
    save_registry(reg)
    return entry


if __name__ == "__main__":                            # `python -m src.model` trains v1 from the command line
    print(json.dumps(run_training(), indent=2))
