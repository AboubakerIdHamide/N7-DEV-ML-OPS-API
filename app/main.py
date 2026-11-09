"""Validated credit-default scoring API with Prometheus request metrics."""

import os
from functools import lru_cache
from time import perf_counter

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.data import FEATURE_COLUMNS


MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-classifier")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@champion")
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Prediction requests by result", ["status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent producing a default prediction"
)

app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Educational API using the public UCI credit-default dataset. "
        "Predictions are not financial advice or lending decisions."
    ),
    version="0.1.0",
)


class CreditApplication(BaseModel):
    limit_bal: float = Field(gt=0, allow_inf_nan=False, examples=[50000.0])
    pay_0: int = Field(ge=-2, le=9, examples=[-1])
    pay_2: int = Field(ge=-2, le=9, examples=[0])
    pay_3: int = Field(ge=-2, le=9, examples=[0])
    pay_4: int = Field(ge=-2, le=9, examples=[0])
    pay_5: int = Field(ge=-2, le=9, examples=[0])
    pay_6: int = Field(ge=-2, le=9, examples=[0])
    bill_amt1: float = Field(allow_inf_nan=False, examples=[12000.0])
    bill_amt2: float = Field(allow_inf_nan=False, examples=[11000.0])
    bill_amt3: float = Field(allow_inf_nan=False, examples=[10500.0])
    bill_amt4: float = Field(allow_inf_nan=False, examples=[9800.0])
    bill_amt5: float = Field(allow_inf_nan=False, examples=[9300.0])
    bill_amt6: float = Field(allow_inf_nan=False, examples=[8700.0])
    pay_amt1: float = Field(ge=0, allow_inf_nan=False, examples=[1500.0])
    pay_amt2: float = Field(ge=0, allow_inf_nan=False, examples=[1200.0])
    pay_amt3: float = Field(ge=0, allow_inf_nan=False, examples=[1000.0])
    pay_amt4: float = Field(ge=0, allow_inf_nan=False, examples=[900.0])
    pay_amt5: float = Field(ge=0, allow_inf_nan=False, examples=[850.0])
    pay_amt6: float = Field(ge=0, allow_inf_nan=False, examples=[800.0])


@lru_cache(maxsize=1)
def get_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    try:
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            "The registered champion model is unavailable; run the training pipeline first."
        ) from exc


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness probe: the API process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
def ready() -> dict[str, str]:
    """Readiness probe: the registered model can be loaded."""
    try:
        get_model()
        return {"status": "ready"}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/schema", tags=["prediction"])
def schema() -> dict[str, object]:
    """Describe the fields and output classes expected by this API."""
    return {
        "input_fields": [field.lower() for field in FEATURE_COLUMNS],
        "output_classes": {"0": "no_default", "1": "default"},
        "data_note": (
            "UCI public dataset; the record ID and demographic fields are excluded "
            "from the selected training data and prediction inputs."
        ),
    }


@app.post("/predict", tags=["prediction"])
def predict(application: CreditApplication) -> dict[str, object]:
    start = perf_counter()
    try:
        model = get_model()
        input_values = application.model_dump()
        row = pd.DataFrame(
            [{column: input_values[column.lower()] for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        prediction = int(model.predict(row)[0])
        probability_default = float(model.predict_proba(row)[0][1])
        PREDICTION_REQUESTS.labels(status="success").inc()
        return {
            "prediction": "default" if prediction == 1 else "no_default",
            "default_probability": probability_default,
            "model": MODEL_NAME,
        }
    except Exception as exc:
        PREDICTION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.observe(perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
