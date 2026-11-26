# NeuroTrace

NeuroTrace classifies scanned spiral and meander drawings from the HandPD
dataset as belonging to the control or Parkinson's group. It demonstrates an
image ML pipeline with offline training, a FastAPI prediction service,
a Streamlit dashboard, monitoring and versioned retraining.

- GitHub — https://github.com/ambabazi/ml-pipeline-summative
- API docs — https://neurotrace-api-it4i.onrender.com/docs
- Dashboard — local/Docker only at http://localhost:8501; not publicly hosted
- YouTube demo — pending upload
- YouTube script and screen cues — [docs/youtube-demo-script.md](docs/youtube-demo-script.md)

## Run locally

Use Python 3.12 from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. By default, the dashboard connects to the
public Render API, so uploaded images and retraining requests go to that
service. For a self-contained local demo, use the Docker deployment below.
To run the dashboard with a local API without Docker, run
`uvicorn api.main:app --port 8000` in one terminal and
`API_URL=http://localhost:8000 streamlit run dashboard/app.py` in another.

For notebook work, install `jupyterlab matplotlib seaborn tensorflow-cpu` in the
same environment. Open `notebook/neurotrace.ipynb` with the notebook directory
as its working directory. The prepared train and test data are in `data/`.
The raw dataset source is [HandPD](https://www2.fc.unesp.br/~papa/pub/datasets/Handpd/).
`scripts/split_data.py` documents the participant-grouped split.

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
must also keep other drawings from evaluation participants out of training. The model history records the starting version and the
outcome. The candidate is promoted when its ROC-AUC on the shared evaluation
set is within 0.02 of the active model or higher. Because this set is used
for promotion, an independent future cohort would give a stronger final check.

On the shared 152-image evaluation set from 19 held-out participants, the
original production model v1 scored accuracy 0.776, precision 0.922, recall
0.783, specificity 0.750, F1 0.847 and ROC-AUC 0.798. Retrained model v2,
selected as active in the registry, scored 0.789, 0.923, 0.800, 0.750, 0.857
and 0.805, respectively. This evaluation set was also used for the promotion
decision, so these results are not an independent final estimate of
generalization. The notebook separately compares an engineered-feature
Random Forest with a CNN and includes grouped development comparisons with
Extra Trees and logistic regression. This project is an educational prototype
for classifying drawings.

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

Run `docker compose up -d --build`, then open the dashboard at
`http://localhost:8501` and the API docs at `http://localhost:8080/docs`.
Compose runs Streamlit, Nginx, and the API. Named volumes retain the SQLite
database and versioned model artifacts across container replacements. Use
one API replica for the retraining demo because job status is held in that
process. On Render, files written to the container are temporary unless
the service uses persistent storage.

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
Locust, then run the following from the repository root. Repeat with replica
counts 2 and 4, changing the CSV prefix each time. The current API can reject some images
in the original workload. Report these responses separately when comparing
new runs to the historical results.

```bash
docker compose up -d --build --scale api=1
curl -fsS http://localhost:8080/health
locust -f monitoring/locustfile.py --headless \
  --host http://localhost:8080 -u 100 -r 10 -t 60s \
  --csv /tmp/neurotrace_containers_1
```