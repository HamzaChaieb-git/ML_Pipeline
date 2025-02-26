import pytest
import pandas as pd
import numpy as np
import os
import mlflow
import tempfile
import xgboost as xgb
from model_training import train_model
from model_evaluation import evaluate_model
from sklearn.preprocessing import LabelEncoder


# Updated MLflow tracking fixture
@pytest.fixture(autouse=True)
def setup_mlflow_tracking(tmp_path):
    """Set up MLflow tracking for tests with a properly initialized experiment."""
    # Create a directory for MLflow tracking
    mlruns_dir = tmp_path / "mlruns"
    os.makedirs(mlruns_dir, exist_ok=True)

    # Set MLflow tracking URI to the temporary directory
    tracking_uri = f"file://{mlruns_dir}"
    mlflow.set_tracking_uri(tracking_uri)

    # Create a test experiment
    experiment_name = "test_experiment"
    try:
        # Check if experiment exists first
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(mlruns_dir, experiment_name),
            )
        else:
            experiment_id = experiment.experiment_id

        # Set as active experiment
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Failed to create MLflow experiment: {e}")

    # Make directories for artifacts that the tests might try to use
    artifacts_dir = tmp_path / "artifacts"
    models_dir = artifacts_dir / "models"
    evaluation_dir = artifacts_dir / "evaluation"
    training_dir = artifacts_dir / "training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    # Create a sample MLflow run to track artifacts under
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id
    ) as run:
        mlflow.log_param("dummy", "value")

    yield

    # No cleanup needed for tmp_path as pytest handles it


@pytest.fixture
def trained_model_and_data(tmp_path):
    X_train = pd.DataFrame(
        {
            "Total day minutes": [120.5, 150.3],
            "Customer service calls": [3, 2],
            "International plan": ["no", "yes"],  # Object dtype (categorical)
            "Total intl minutes": [10.2, 8.5],
            "Total intl calls": [5, 4],
            "Total eve minutes": [200.0, 180.5],
            "Number vmail messages": [0, 5],
            "Voice mail plan": ["no", "yes"],  # Object dtype (categorical)
        }
    )
    y_train = np.array([0, 1])

    # Encode categorical columns
    le_international = LabelEncoder()
    le_voice = LabelEncoder()
    X_train["International plan"] = le_international.fit_transform(
        X_train["International plan"]
    )
    X_train["Voice mail plan"] = le_voice.fit_transform(X_train["Voice mail plan"])

    model = train_model(X_train, y_train, model_version="test_1.0")
    return model, X_train, y_train


def test_evaluate_model_metrics(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    assert "roc_auc" in metrics, "ROC AUC should be in metrics"
    assert "log_loss" in metrics, "Log loss should be in metrics"
    assert (
        metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1
    ), "Accuracy should be between 0 and 1"
    assert (
        metrics["roc_auc"] >= 0 and metrics["roc_auc"] <= 1
    ), "ROC AUC should be between 0 and 1"


def test_evaluate_model_artifacts(trained_model_and_data, tmp_path):
    model, X_test, y_test = trained_model_and_data

    # Set MLflow artifact root to a temporary directory
    temp_artifacts = tmp_path / "artifacts"
    os.makedirs(temp_artifacts, exist_ok=True)

    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict), "Metrics should still be generated"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
