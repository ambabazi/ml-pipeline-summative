"""NeuroTrace dashboard: monitoring, data insights, prediction, upload and retraining."""
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://neurotrace-api-it4i.onrender.com").rstrip("/")  # the deployed FastAPI service
ROOT = Path(__file__).resolve().parent.parent          # project root (dashboard/..)
FEATURES_CSV = ROOT / "reports" / "features.csv"       # features of every train/test image, exported by the notebook
LOAD_TESTS = ROOT / "reports" / "load_tests"           # Locust CSV results
COLORS = {"healthy": "#2a9d8f", "parkinson": "#e76f51"}  # one colour per class, used in every chart

st.set_page_config(page_title="NeuroTrace", page_icon="🌀", layout="wide")


def api(method, path, timeout=120, **kwargs):
    """Call the API and return JSON, or None (with an error message) if it fails."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        r.raise_for_status()                           # turn 4xx/5xx into an exception
        return r.json()
    except requests.HTTPError as error:
        if error.response.status_code in {400, 409}:
            try:
                detail = error.response.json().get("detail")
            except ValueError:
                detail = None
            st.warning(detail or "The API rejected the request.")
        else:
            st.error(f"API error on {path}: {error}")
        return None
    except requests.RequestException as e:
        st.error(f"API error on {path}: {e}")
        return None


def fmt_uptime(seconds):
    """123456 -> '1d 10h 17m'"""
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    return f"{d}d {h}h {rem // 60}m" if d else f"{h}h {rem // 60}m {rem % 60}s"


# ---------------- Header ----------------
st.title("🌀 NeuroTrace")
st.caption("Classifying Parkinsonian patterns in hand-drawn spirals and meanders - "
           "**a research screening prototype, not a diagnostic tool.**")

tab_overview, tab_predict, tab_insights, tab_train, tab_monitor = st.tabs(
    ["📊 Overview", "🔍 Predict", "📈 Data insights", "🔁 Train & retrain", "⚡ Monitoring"])

# ---------------- 1. Overview: uptime + live model ----------------
with tab_overview:
    st.info("The free cloud server sleeps when idle - the first request can take up to a minute.")
    health = api("GET", "/health")
    registry = api("GET", "/models")
    if health:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API status", "🟢 Online" if health["status"] == "ok" else "🔴 Down")
        c2.metric("Uptime", fmt_uptime(health["uptime_seconds"]))
        c3.metric("Live model", f"v{health['model_version']}")
        c4.metric("Model", health["model_status"])
    if registry and registry["versions"]:
        live = next(v for v in registry["versions"] if v["version"] == registry["active"])
        st.subheader(f"Live model v{live['version']} - test-set performance")
        m = live["metrics"]
        cols = st.columns(6)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]):
            col.metric(key.replace("_", "-").upper() if key == "roc_auc" else key.capitalize(), f"{m[key]:.3f}")
        st.caption(f"Trained {live['trained_at'][:19].replace('T', ' ')} UTC on {live['n_train_images']} images "
                   f"({live['n_uploaded_images']} uploaded). Scored on the shared evaluation set "
                   "(152 images, 19 people) every time.")

# ---------------- 2. Predict one drawing ----------------
with tab_predict:
    st.subheader("Analyse one drawing")
    image = st.file_uploader("Upload a blue-ink spiral or meander on a dark printed guide (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if image:
        left, right = st.columns([1, 1])
        left.image(image, caption=image.name, width="stretch")
        if right.button("Analyse drawing", type="primary"):
            with st.spinner("Extracting features and predicting..."):
                res = api("POST", "/predict", files={"file": (image.name, image.getvalue(), image.type)})
            if res:
                is_pd = res["prediction"] == "parkinson"
                right.markdown(f"### {'🟠 Parkinsonian pattern' if is_pd else '🟢 Healthy pattern'}")
                right.metric("Model score for this class", f"{res['confidence']:.0%}")
                right.progress(res["probability_parkinson"], text=f"Parkinson’s-group score = {res['probability_parkinson']:.2f}")
                right.caption(f"Model v{res['model_version']} · {res['disclaimer']}")
                # Compare this drawing's features with the typical healthy / Parkinson's drawing
                if FEATURES_CSV.exists():
                    med = pd.read_csv(FEATURES_CSV).groupby("label").median(numeric_only=True).T
                    med = med.loc[med.index.intersection(res["features"])]
                    med["this drawing"] = pd.Series(res["features"])
                    st.markdown("**How this drawing compares with typical drawings (medians)**")
                    st.dataframe(med[["this drawing", "healthy", "parkinson"]].round(3), width="stretch")

# ---------------- 3. Data insights: 3 features, interpreted ----------------
with tab_insights:
    if not FEATURES_CSV.exists():
        st.warning("reports/features.csv not found - export it from the notebook.")
    else:
        feats = pd.read_csv(FEATURES_CSV)
        c1, c2 = st.columns([1, 2])
        counts = feats.groupby(["label", "drawing"]).size().reset_index(name="images")
        c1.plotly_chart(px.bar(counts, x="label", y="images", color="drawing", barmode="group",
                               title="Images per class"), width="stretch")
        c2.markdown(
            "**Class imbalance:** about 4 Parkinson's drawings for every healthy one (74 vs 18 people). "
            "Accuracy alone would be misleading - a model that always says *Parkinson's* scores ~79% - "
            "so the models are judged on recall, specificity, F1 and ROC-AUC, with class weights in training.")

        stories = {
            "fragments": ("Stroke fragmentation",
                          "Connected ink regions of at least five pixels. The medians were 13 for "
                          "control drawings and 20 for Parkinson’s-group drawings."),
            "ink_ratio": ("Ink-to-template area",
                          "Detected blue ink pixels divided by dark template pixels. The medians "
                          "were 1.57 for controls and 1.16 for the Parkinson’s group."),
            "deviation": ("Deviation from the template",
                          "Average pen-to-template distance. The medians were 3.14 and 3.72 pixels "
                          "for the control and Parkinson’s groups, respectively."),
        }
        for feature, (title, story) in stories.items():
            c1, c2 = st.columns([2, 1])
            fig = px.box(feats, x="drawing", y=feature, color="label", color_discrete_map=COLORS, title=title)
            c1.plotly_chart(fig, width="stretch")
            c2.markdown(f"#### {title}")
            c2.write(story)
        st.caption("The groups overlap on all three measures. Ink colour and scan quality can change the measurements.")

# ---------------- 4. Upload data + retrain ----------------
with tab_train:
    st.subheader("1 · Upload labelled drawings (bulk)")
    label = st.radio("These drawings are from:", ["healthy", "parkinson"], horizontal=True)
    batch = st.file_uploader("Select many images at once", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if batch and st.button(f"Upload {len(batch)} image(s) as '{label}'"):
        with st.spinner("Uploading..."):
            res = api("POST", "/upload", timeout=300, data={"label": label},
                      files=[("files", (f.name, f.getvalue(), f.type)) for f in batch])
        if res:
            st.write(f"New images saved in this upload: {res['saved']}.")
            if res["rejected"]:
                st.warning(f"Unsupported images: {res['rejected']}")

            if res.get("duplicates"):
                st.info(f"Already saved: {res['duplicates']}")
            if res.get("already_used"):
                st.info(f"Already used by the active model: {res['already_used']}")
            if res.get("fixed_set"):
                st.warning(f"Already in the bundled training or evaluation set: {res['fixed_set']}")

    st.subheader("2 · Retrain the model")
    st.write("Accepted images are saved in SQLite. Retraining loads the active forest and "
             "adds 50 trees using the base training images and uploads the active model has yet to use. "
             "The existing and candidate models are compared on the shared evaluation set.")
    if st.button("🔁 Trigger retraining", type="primary"):
        if api("POST", "/retrain"):
            with st.status("Retraining in progress...", expanded=True) as status:
                start = time.time()
                while True:
                    state = api("GET", "/retrain/status") or {}
                    if state.get("status") != "running":
                        break
                    status.update(label=f"Retraining in progress... {int(time.time() - start)} s")
                    time.sleep(5)                       # check again every 5 seconds
                if state.get("status") == "completed":
                    r = state["result"]
                    verdict = "✅ promoted to live" if r["promoted"] else "⛔ kept offline (quality gate)"
                    status.update(label=f"Model v{r['version']} trained - {verdict}", state="complete")
                    st.caption(f"Started from model v{r.get('parent_version')} · {r['n_uploaded_images']} new uploads used in this run")
                    st.json(r["metrics"])
                else:
                    status.update(label=f"Retraining failed: {state.get('error')}", state="error")

    batch_status = api("GET", "/uploads/status")
    if batch_status is not None:
        stored, used, pending = st.columns(3)
        stored.metric("Stored uploads", batch_status["stored"])
        used.metric("Stored uploads used by active model", batch_status["used"])
        pending.metric("Pending uploads", batch_status["pending"])

    st.subheader("3 · Model versions")
    registry = api("GET", "/models")
    if registry and registry["versions"]:
        rows = [{"version": f"v{v['version']}" + (" (live)" if v["version"] == registry["active"] else ""),
                 "trained_at": v["trained_at"][:19].replace("T", " "), "train images": v["n_train_images"],
                 "new uploads used": v["n_uploaded_images"], "promoted": v["promoted"], **v["metrics"]}
                for v in registry["versions"]]
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch", hide_index=True)
        long = table.melt(id_vars="version", value_vars=["accuracy", "recall", "specificity", "f1", "roc_auc"],
                          var_name="metric", value_name="score")
        st.plotly_chart(px.bar(long, x="metric", y="score", color="version", barmode="group",
                               title="Version comparison on the same test set", range_y=[0, 1]),
                        width="stretch")

# ---------------- 5. Monitoring: live stats + load tests ----------------
with tab_monitor:
    stats = api("GET", "/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests served", stats["requests"])
        c2.metric("Predictions", stats["predictions"])
        c3.metric("Errors", stats["errors"])
        c4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
        st.caption(f"Counters since the API container started ({stats['container']}).")

    st.subheader("Recorded Locust runs before upload validation")
    rows = []
    for csv in sorted(LOAD_TESTS.glob("*_stats.csv")):
        agg = pd.read_csv(csv).query("Name == 'Aggregated'").iloc[0]
        rows.append({"setup": csv.stem.replace("_stats", "").replace("_", " "),
                     "requests": int(agg["Request Count"]), "failures": int(agg["Failure Count"]),
                     "median_ms": agg["50%"], "p95_ms": agg["95%"], "req_per_s": round(agg["Requests/s"], 1)})
    if rows:
        lt = pd.DataFrame(rows)
        st.dataframe(lt, width="stretch", hide_index=True)
        local = lt[lt.setup.str.startswith("containers")]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(local, x="setup", y=["median_ms", "p95_ms"], barmode="group",
                               title="Response time vs number of containers"), width="stretch")
        c2.plotly_chart(px.bar(local, x="setup", y="req_per_s", title="Throughput vs number of containers"),
                        width="stretch")
