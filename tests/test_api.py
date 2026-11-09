from fastapi.testclient import TestClient

from app.main import app


class FakeModel:
    def predict(self, rows):
        return [1]

    def predict_proba(self, rows):
        return [[0.2, 0.8]]


def valid_application():
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


def test_prediction_endpoint_uses_registered_model(monkeypatch):
    monkeypatch.setattr("app.main.get_model", lambda: FakeModel())
    response = TestClient(app).post("/predict", json=valid_application())
    assert response.status_code == 200
    assert response.json()["prediction"] == "default"
    assert response.json()["default_probability"] == 0.8


def test_prediction_rejects_ratio_outside_valid_range():
    application = valid_application()
    application["pay_0"] = 10
    response = TestClient(app).post("/predict", json=application)
    assert response.status_code == 422


def test_prediction_rejects_missing_required_field():
    application = valid_application()
    del application["limit_bal"]
    response = TestClient(app).post("/predict", json=application)
    assert response.status_code == 422
