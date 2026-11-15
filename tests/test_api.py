"""HTTP-level tests for request validation and prediction API behavior."""

from fastapi.testclient import TestClient

from app.main import app


class FakeModel:
    """Small stand-in for the registered classifier used by API tests."""

    def predict(self, rows):
        """Return one default-class label for each test request row."""
        return [1]

    def predict_proba(self, rows):
        """Return probabilities ordered as no-default then default."""
        return [[0.2, 0.8]]


class FakeNoDefaultModel(FakeModel):
    """Fake classifier that predicts the no-default class."""

    def predict(self, rows):
        """Return the no-default label for the test request."""
        return [0]


def valid_application():
    """Return one request body containing all required fields with valid values."""
    return {
        "limit_bal": 50000,
        "pay_0": -1,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 12000,
        "bill_amt2": 11000,
        "bill_amt3": 10500,
        "bill_amt4": 9800,
        "bill_amt5": 9300,
        "bill_amt6": 8700,
        "pay_amt1": 1500,
        "pay_amt2": 1200,
        "pay_amt3": 1000,
        "pay_amt4": 900,
        "pay_amt5": 850,
        "pay_amt6": 800,
    }


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_when_model_is_available(monkeypatch):
    monkeypatch.setattr("app.main.get_model", lambda: FakeModel())
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_endpoint_when_model_is_unavailable(monkeypatch):
    def unavailable_model():
        """Simulate MLflow failing to load the registered model."""
        raise RuntimeError("model is not loaded")

    monkeypatch.setattr("app.main.get_model", unavailable_model)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "model is not loaded"


def test_schema_lists_only_selected_model_inputs():
    response = TestClient(app).get("/schema")
    assert response.status_code == 200
    assert len(response.json()["input_fields"]) == 19
    assert "id" not in response.json()["input_fields"]
    assert "sex" not in response.json()["input_fields"]
    assert "age" not in response.json()["input_fields"]


def test_prediction_endpoint_uses_registered_model(monkeypatch):
    monkeypatch.setattr("app.main.get_model", lambda: FakeModel())
    response = TestClient(app).post("/predict", json=valid_application())
    assert response.status_code == 200
    assert response.json()["prediction"] == "default"
    assert response.json()["default_probability"] == 0.8


def test_prediction_can_return_no_default(monkeypatch):
    monkeypatch.setattr("app.main.get_model", lambda: FakeNoDefaultModel())
    response = TestClient(app).post("/predict", json=valid_application())
    assert response.status_code == 200
    assert response.json()["prediction"] == "no_default"


def test_prediction_rejects_ratio_outside_valid_range():
    application = valid_application()
    application["pay_0"] = 10
    response = TestClient(app).post("/predict", json=application)
    assert response.status_code == 422


def test_prediction_rejects_negative_payment_amount():
    application = valid_application()
    application["pay_amt1"] = -1
    response = TestClient(app).post("/predict", json=application)
    assert response.status_code == 422


def test_prediction_rejects_missing_required_field():
    application = valid_application()
    del application["limit_bal"]
    response = TestClient(app).post("/predict", json=application)
    assert response.status_code == 422
