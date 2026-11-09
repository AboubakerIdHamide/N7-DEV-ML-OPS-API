# Project 2: Credit Default Prediction API

An educational MLOps project for the Project 2 brief. It trains a binary classifier on the public UCI Default of Credit Card Clients dataset, tracks experiments and registers a model in MLflow, and serves predictions through FastAPI. Docker Compose runs the tracking server, trainer, API, and Prometheus.

## Prediction scope

For each client record, the model uses the credit limit and account repayment history to estimate the probability of the dataset's `default payment next month` target being positive. The API returns a probability and a class label; it does not make a lending decision.

The first feature set contains only the credit limit, six monthly repayment-status fields, six bill amounts, and six payment amounts. The source record ID and demographic fields are excluded before the local cache is written and are not model inputs.

The source data was collected in Taiwan in 2005. It is public research data and is licensed under CC BY 4.0. Cite the source as: Yeh, I. (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, DOI: <https://doi.org/10.24432/C55S3H>. The dataset and resulting model are for coursework only, not current lending decisions.

## Current stage

The initial pipeline uses the actual UCI dataset rather than generated synthetic records. On the first training run, the trainer downloads the dataset through `ucimlrepo`, selects the approved fields, and caches that selected CSV in a Docker named volume. It is not committed to Git. Data versioning with DVC and a configured remote are the next MLOps stage.

## Run the end-to-end pipeline with Docker

You need Docker with the Compose plugin and network access for the first dataset download. From this directory:

```bash
docker compose up -d mlflow
docker compose run --rm trainer
docker compose up --build api prometheus
```

The training run compares Logistic Regression with Random Forest, logs default-class precision, recall, F1 and ROC-AUC to MLflow, and registers the model with the best default-class F1 score under the `champion` alias.

- API docs: <http://localhost:8000/docs>
- API schema: <http://localhost:8000/schema>
- API liveness: <http://localhost:8000/health>
- API readiness: <http://localhost:8000/ready>
- MLflow UI: <http://localhost:5000>
- Prometheus: <http://localhost:9090>
- API metrics: <http://localhost:8000/metrics>

## Try a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"limit_bal":50000,"pay_0":-1,"pay_2":0,"pay_3":0,"pay_4":0,"pay_5":0,"pay_6":0,"bill_amt1":12000,"bill_amt2":11000,"bill_amt3":10500,"bill_amt4":9800,"bill_amt5":9300,"bill_amt6":8700,"pay_amt1":1500,"pay_amt2":1200,"pay_amt3":1000,"pay_amt4":900,"pay_amt5":850,"pay_amt6":800}'
```

The API validates the required fields and returns `default_probability` and a `prediction` of `default` or `no_default`.

## Run locally without Docker

Create and activate a virtual environment, install requirements, and start MLflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal using that environment:

```bash
python -m app.train
uvicorn app.main:app --reload
```

## Repository map

```text
app/data.py       fetch, select, cache, and validate the UCI data
app/train.py      train, compare, track, and register candidate models
app/main.py       validated prediction API and Prometheus metrics
tests/            API input and response checks
Dockerfile        shared image for MLflow, training, and API services
docker-compose.yml
prometheus.yml
```

## Work remaining against the module brief

- Version the selected dataset with DVC and configure a usable remote.
- Add GitHub Actions to run checks on commits or pull requests.
- Add data-preparation and model-validation tests with a quality gate.
- Add a monitoring dashboard and finish the deployment guide.
- Prepare the required technical report, presentation, and clean-clone demonstration.
