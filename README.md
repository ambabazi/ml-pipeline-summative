# NeuroTrace

NeuroTrace classifies scanned spiral and meander drawings from the HandPD
dataset as belonging to the control or Parkinson's group. It demonstrates an
image ML pipeline with offline training, a FastAPI prediction service,
a Streamlit dashboard, monitoring and versioned retraining.

- GitHub — https://github.com/ambabazi/ml-pipeline-summative
- API docs — https://neurotrace-api-it4i.onrender.com/docs
- Dashboard — http://localhost:8501 after starting the local or Docker deployment
- YouTube demo — pending upload

## Run locally

Use Python 3.12 from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. The dashboard connects to the Render API by default.
For a local API, run `uvicorn api.main:app --port 8000` in one terminal and
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
model score. Use labelled images from `data/retrain_batch/` for a bulk retrain
demo. Upload one label at a time and click Trigger retraining.

Accepted images and their labels are saved in `data/uploads/uploads.sqlite3`.
An exact duplicate is skipped. Retraining loads the active pretrained forest
and adds 50 trees using the base images and accumulated uploads. Earlier
trees are retained. The model history records the starting version and the
outcome. The candidate is promoted when its ROC-AUC on the shared evaluation
set is within 0.02 of the active model or higher. Because this set is used
for promotion, an independent future cohort would give a stronger final check.

On 152 test images from 19 participants, the deployed v1 model had accuracy
0.776, precision 0.922, recall 0.783, specificity 0.750, F1 0.847 and ROC-AUC
0.798. The notebook compares a separate Random Forest experiment with a CNN and
includes a grouped development comparison with Extra Trees and logistic regression.
This project is an educational prototype for classifying drawings.

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
3. Upload images from `data/retrain_batch/healthy/` with the healthy label,
   then repeat with `data/retrain_batch/parkinson/`. Show the saved counts.
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