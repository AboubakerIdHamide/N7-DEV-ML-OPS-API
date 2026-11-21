"""Tests for model quality checks and promotion into the champion alias."""

from types import SimpleNamespace

import pytest

from app.train import _promote_candidate, _validate_minimum_default_f1


def candidate_result(f1_default):
    """Build the result fields promotion needs from one logged candidate."""
    return {"f1_default": f1_default, "run_id": "run-123"}


@pytest.mark.parametrize("threshold", [0.0, 0.45, 1.0])
def test_minimum_f1_accepts_values_inside_range(threshold):
    assert _validate_minimum_default_f1(threshold) == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_minimum_f1_rejects_values_outside_range(threshold):
    with pytest.raises(ValueError, match="MIN_DEFAULT_F1"):
        _validate_minimum_default_f1(threshold)


def test_quality_gate_does_not_register_or_change_champion(monkeypatch):
    def fail_if_registered(**kwargs):
        """Fail if a below-threshold candidate reaches the model registry."""
        pytest.fail("the rejected candidate must not be registered")

    monkeypatch.setattr("app.train.mlflow.register_model", fail_if_registered)

    with pytest.raises(RuntimeError, match="champion was not changed"):
        _promote_candidate(candidate_result(0.449), "credit-risk-classifier", 0.45)


def test_quality_gate_registers_and_assigns_passing_candidate(monkeypatch):
    alias_updates = []
    registered = SimpleNamespace(version="3")

    def register_model(model_uri, name):
        """Capture the model location and return a fake registry version."""
        assert model_uri == "runs:/run-123/model"
        assert name == "credit-risk-classifier"
        return registered

    class FakeMlflowClient:
        """Capture the alias update without contacting an MLflow server."""

        def set_registered_model_alias(self, model_name, alias, version):
            alias_updates.append((model_name, alias, version))

    monkeypatch.setattr("app.train.mlflow.register_model", register_model)
    monkeypatch.setattr("app.train.MlflowClient", FakeMlflowClient)

    result = _promote_candidate(
        candidate_result(0.45), "credit-risk-classifier", 0.45
    )

    assert result is registered
    assert alias_updates == [("credit-risk-classifier", "champion", "3")]
