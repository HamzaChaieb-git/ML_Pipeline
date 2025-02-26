import pytest
import pandas as pd
import numpy as np
import os
import mlflow
import tempfile
import xgboost as xgb
from model_training import train_model


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
    training_dir = artifacts_dir / "training"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    # Create a sample MLflow run to track artifacts under
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id
    ) as run:
        mlflow.log_param("dummy", "value")

    yield

    # No cleanup needed for tmp_path as pytest handles it


# Fixture for sample training data
@pytest.fixture
def sample_train_data():
    X_train = pd.DataFrame(
        {
            "Total day minutes": [120.5, 150.3],
            "Customer service calls": [3, 2],
            "International plan": [0, 1],  # Already encoded
            "Total intl minutes": [10.2, 8.5],
            "Total intl calls": [5, 4],
            "Total eve minutes": [200.0, 180.5],
            "Number vmail messages": [0, 5],
            "Voice mail plan": [0, 1],  # Already encoded
        }
    )
    y_train = np.array([0, 1])
    return X_train, y_train


def test_train_model_returns_model(sample_train_data):
    X_train, y_train = sample_train_data
    model = train_model(X_train, y_train, model_version="test_1.0")
    assert isinstance(model, xgb.XGBClassifier), "Model should be an XGBClassifier"
    assert model is not None, "Model should not be None"


def test_train_model_predicts(sample_train_data):
    X_train, y_train = sample_train_data
    model = train_model(X_train, y_train, model_version="test_1.0")
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train), "Predictions should match input length"
    assert all(p in [0, 1] for p in predictions), "Predictions should be binary"


def test_train_model_empty_data():
    with pytest.raises(ValueError, match="Training data or labels cannot be empty"):
        train_model(pd.DataFrame(), np.array([]), model_version="test_1.0")
