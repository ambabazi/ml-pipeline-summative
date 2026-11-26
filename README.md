# NeuroTrace

NeuroTrace classifies scanned spiral and meander drawings from the HandPD
dataset as belonging to the control or Parkinson's group. It demonstrates an
image ML pipeline with offline training, a FastAPI prediction service,
a Streamlit dashboard, monitoring and versioned retraining.

- GitHub — https://github.com/ambabazi/ml-pipeline-summative
- API docs — https://neurotrace-api-it4i.onrender.com/docs#/
- dashboard — (https://ml-pipeline-summative.streamlit.app/)
- YouTube demo — link pending recording

## Run locally

Install [Python 3.12](https://www.python.org/downloads/) and
[Git](https://git-scm.com/downloads). Clone the repository or extract the
submission ZIP, then open a terminal in the project root.

```text
git clone https://github.com/ambabazi/ml-pipeline-summative.git
cd ml-pipeline-summative
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r dashboard/requirements.txt
```

You can activate the environment with `.\.venv\Scripts\Activate.ps1`.
The commands below call the virtual environment's Python directly, so
they also work when PowerShell blocks activation scripts.

### Linux and macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r dashboard/requirements.txt
```

The root requirements install FastAPI, Uvicorn, OpenCV and the model
dependencies. The dashboard requirements install Streamlit and its charting
and HTTP dependencies. SQLite is included with Python.

### Start a local API and dashboard

Use two terminals from the project root. On Linux or macOS, activate the
environment in both terminals.

| Terminal | Windows PowerShell | Linux or macOS |
| --- | --- | --- |
| 1 · API | `.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000` | `python -m uvicorn api.main:app --port 8000` |
| 2 · Dashboard URL setting | `$env:API_URL = "http://127.0.0.1:8000"` | `export API_URL=http://127.0.0.1:8000` |
| 2 · Start dashboard | `.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py` | `python -m streamlit run dashboard/app.py` |

Open the [dashboard](http://localhost:8501), [API health](http://127.0.0.1:8000/health)
and [API docs](http://127.0.0.1:8000/docs). Without the `API_URL` override,
Streamlit uses the deployed Render API, including for upload and retraining.

### Run the notebook

The executed notebook is `notebook/neurotrace.ipynb`. To rerun it, install
JupyterLab, Matplotlib and seaborn in the virtual environment:

```text
python -m pip install jupyterlab matplotlib seaborn
```

Install `tensorflow-cpu` on supported Linux or Windows setups, or `tensorflow`
on supported macOS setups. Check the platform and Python version first
if pip cannot resolve a wheel. This dependency is needed for the CNN
notebook cells. [TensorFlow installation guide](https://www.tensorflow.org/install/pip)

From the project root, run `cd notebook` and `python -m jupyterlab`. On
Windows, either activate the environment first or use
`..\.venv\Scripts\python.exe -m jupyterlab` from that folder. The prepared
data is under `data/`. The [HandPD source](https://www2.fc.unesp.br/~papa/pub/datasets/Handpd/)
and `scripts/split_data.py` document acquisition and the split.

## Where to find things

| Path | Contents |
| --- | --- |
| `notebook/` | Executed experiments, feature plots and model evaluation |
| `data/` | Prepared training, test and original retraining drawings |
| `src/` | Image preprocessing, prediction, retraining and SQLite storage |
| `models/` | Saved models and active-version registry |
| `api/` and `dashboard/` | FastAPI endpoints and Streamlit interface |
| `monitoring/` and `reports/` | Locust workload, benchmark CSVs and figures |

## How the pipeline works

OpenCV separates the pen trace from the printed template and extracts nine
measurements. The API loads the active Random Forest listed in
`models/registry.json`. Upload one JPG or PNG in Predict to see its class and
model score. For retraining, upload a labelled batch that has not already been
used to train or evaluate the active model, then click Trigger retraining.
The existing `data/retrain_batch/` files were already used to produce v2; use
the isolated v1 replay described below if demonstrating with those files again.

Accepted images and their labels are saved in `data/uploads/uploads.sqlite3`.
The API skips exact duplicates, including copies of bundled base or evaluation
images and drawings already used by the active model. Retraining loads the
active forest and adds 50 trees using the base images and pending uploads.
Pending uploads become used when a candidate is promoted. A rejected
candidate leaves its batch pending. The dashboard shows newly saved, stored,
used, and pending counts. File fingerprints check exact copies. Uploaders
must also keep other drawings from evaluation participants out of training.
The model history records the starting version and the outcome. The candidate is promoted when its ROC-AUC on the shared evaluation
set is within 0.02 of the active model or higher. Because this set is used
for promotion, an independent future cohort would give a stronger final check.

### Model results

These results use the same 152 drawings from 19 held-out exam groups.
The HandPD metadata links some participant IDs to exams on both sides
of the split, so these scores measure performance on held-out exams.
A participant-independent evaluation would require a split based on
verified participant IDs and retraining.

| Production model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 · Original forest | 0.7763 | 0.9216 | 0.7833 | 0.7500 | 0.8468 | 0.7982 |
| v2 · Forest selected for deployment | 0.7895 | 0.9231 | 0.8000 | 0.7500 | 0.8571 | 0.8047 |

The live version is visible in the [Render model registry](https://neurotrace-api-it4i.onrender.com/models).
Streamlit rounds the score to three decimal places. The notebook also
compares a separate engineered-feature Random Forest with a CNN and
includes grouped development comparisons with Extra Trees and logistic
regression. NeuroTrace is an educational prototype for classifying drawings.

The current v2 was trained using the 120 images in `data/retrain_batch/`.
Do not present those same files as new training data when retraining v2 again.
For a repeatable demonstration with this batch, use an isolated copy of the
project configured to start from v1, or obtain a new labelled batch that was
not used for training or evaluation.

### Unsupported uploads

The API expects a blue trace over a dark printed spiral or meander guide.
It can reject uploads before classification when the ink and guide masks
fall outside its supported range. The initial check accepted 148 of 152
HandPD test drawings. Four Parkinson's-group drawings were rejected.
Testing on unrelated photos is separate from evaluating the drawing classifier.

## Docker deployment and demo

Install [Docker with Compose v2](https://docs.docker.com/compose/install/).
On Windows or macOS, start Docker Desktop first. On Linux, start Docker
Engine if you use the Compose plugin. From the project root, run:

```text
docker compose config --quiet
docker compose up -d --build --scale api=1
docker compose ps
```

Open the [dashboard](http://localhost:8501), [API health](http://localhost:8080/health)
or [upload status](http://localhost:8080/uploads/status). Compose starts
Streamlit, Nginx and the API. Use one API replica for retraining because
job status is held in that process. Named volumes keep the SQLite data
and model artifacts between runs. An existing model volume can retain an
earlier registry even after rebuilding the image. Use a separate Compose
project for a fresh demo. If a service fails, run
`docker compose logs --tail=100 api nginx dashboard`.
`docker compose down` stops this stack and preserves its volumes.
Render needs a persistent disk to retain runtime changes across deploys.

1. Show the dashboard uptime and live model version.
2. Predict `data/test/healthy/spiral_0102-1.jpg` and compare it with its dataset
   label. Try a blank image or unrelated photo and report what happens.
3. To replay the original 120-image retraining, follow the isolated v1 setup
   in `docs/youtube-demo-script.md` before uploading `data/retrain_batch/`.
   Since the registry already contains versions 1 and 2, this replay creates a
   v3 candidate with v1 as its parent. Otherwise, upload only a new labelled
   batch not already used for training or evaluation.
4. Trigger retraining. Show the parent version, candidate metrics, promotion
   result and active model version. Make one more prediction.
5. Show the three feature plots and explain the saved Locust results.

## Locust flood test

The saved runs predate the input check. They used 100 simulated users for 60 seconds, mixing `/predict` and
`/health`. These are aggregated client response times from the recorded CSVs.

| Setup | Requests | Failures | Median (ms) | p95 (ms) | Requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local, 1 API container | 1,582 | 0 | 2,600 | 3,800 | 26.88 |
| Local, 2 API containers | 2,335 | 0 | 1,400 | 2,300 | 39.56 |
| Local, 4 API containers | 2,675 | 0 | 1,000 | 2,400 | 45.19 |
| Render, cloud run | 248 | 0 | 3,500 | 5,100 | 4.36 |

Four local containers improved median response time and throughput over two.
Two containers had the lowest p95, so the slowest requests did not improve
consistently with more replicas. Render is a separate environment with
different compute and network conditions.

Raw results are in `reports/load_tests/`. To reproduce a local run, install
Locust with `python -m pip install locust`, then run the following from the repository root. Repeat with replica
counts 2 and 4, changing the CSV prefix each time. The current API can reject some images
in the original workload. Report these responses separately when comparing
new runs to the historical results.

```text
docker compose up -d --build --scale api=1
python -m locust -f monitoring/locustfile.py --headless --host http://localhost:8080 -u 100 -r 10 -t 60s --csv neurotrace_containers_1
```

Check [API health](http://localhost:8080/health) before starting Locust.
