"""Train, evaluate, version and retrain the NeuroTrace Random Forest."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.preprocessing import FEATURES, extract_features

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
    """Base training images (v1 data) + every image uploaded through the API."""
    roles = pd.read_csv(DATA / "train_roles.csv")                       # which train image is fit/val/new_batch
    base = [DATA / "train" / f for f in roles[roles.role != "new_batch"].file]  # held-back batch is NOT in the base
    uploads = [p for label in LABELS for p in (DATA / "uploads" / label).glob("*")
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]      # new data users uploaded
    return base + uploads, len(uploads)


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
    MODELS.mkdir(exist_ok=True)
    paths, n_uploads = training_images()
    train_df = image_table(paths)
    test_df = image_table(sorted((DATA / "test").glob("*/*.jpg")))  # same test set every time = fair comparison

    model, params = train(train_df)
    metrics = evaluate(model, test_df)

    reg = load_registry()
    version = len(reg["versions"]) + 1                # next version number
    file = f"neurotrace_rf_v{version}.pkl"
    joblib.dump(model, MODELS / file)                 # save the new model next to the old ones

    current = next((v for v in reg["versions"] if v["version"] == reg["active"]), None)  # the live model's entry
    # Quality gate: only go live if ROC-AUC is not worse than the live model by more than min_auc_drop
    promoted = current is None or metrics["roc_auc"] >= current["metrics"]["roc_auc"] - min_auc_drop

    entry = {"version": version, "file": file, "trained_at": datetime.now(timezone.utc).isoformat(),
             "n_train_images": len(train_df), "n_uploaded_images": n_uploads,
             "params": params, "metrics": metrics, "promoted": promoted}
    reg["versions"].append(entry)                     # keep a record even if not promoted
    if promoted:
        reg["active"] = version                       # switch the live model
    save_registry(reg)
    return entry


if __name__ == "__main__":                            # `python -m src.model` trains v1 from the command line
    print(json.dumps(run_training(), indent=2))