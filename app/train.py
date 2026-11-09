"""Train candidate binary classifiers and record the selected model in MLflow."""

import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_credit_dataset


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk")
    model_name = os.getenv("MODEL_NAME", "credit-risk-classifier")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    dataset = load_credit_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[FEATURE_COLUMNS],
        dataset[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=dataset[TARGET_COLUMN],
    )

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
    }
    results = []

    for model_key, model in candidates.items():
        with mlflow.start_run(run_name=model_key) as run:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)[:, 1]
            metrics = {
                "f1_default": f1_score(y_test, predictions, pos_label=1),
                "precision_default": precision_score(y_test, predictions, pos_label=1),
                "recall_default": recall_score(y_test, predictions, pos_label=1),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            mlflow.log_param("model_type", model_key)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("dataset_rows", len(dataset))
            mlflow.log_param("dataset_id", 350)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset", "UCI-Default-of-Credit-Card-Clients-350")
            mlflow.set_tag("positive_class", "default_next_month")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=infer_signature(x_test, predictions),
                input_example=x_test.head(2),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            results.append((metrics["f1_default"], run.info.run_id, model_key, metrics))

    _, best_run_id, best_model_key, best_metrics = max(results, key=lambda item: item[0])
    registered = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model", name=model_name
    )
    MlflowClient().set_registered_model_alias(model_name, "champion", registered.version)

    print(f"Selected model by default F1: {best_model_key}")
    print(f"Held-out metrics: {best_metrics}")
    print(f"Registered model: models:/{model_name}@champion (version {registered.version})")


if __name__ == "__main__":
    main()
