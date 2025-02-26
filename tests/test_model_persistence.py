import pytest
import os
import mlflow
import tempfile
from model_persistence import save_model, load_model
import xgboost as xgb
import joblib


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

    # Create a sample MLflow run to track artifacts under
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id
    ) as run:
        mlflow.log_param("dummy", "value")

    yield

    # No cleanup needed for tmp_path as pytest handles it


# Fixture for sample model and temporary filename
@pytest.fixture
def sample_model(tmp_path):
    model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False)
    filename = tmp_path / "test_model.joblib"
    return model, str(filename)


def test_save_model(sample_model):
    model, filename = sample_model
    save_model(model, filename)
    assert os.path.exists(filename), "Model file should be saved"


def test_load_model(sample_model):
    model, filename = sample_model
    save_model(model, filename)
    loaded_model = load_model(filename)
    assert isinstance(
        loaded_model, xgb.XGBClassifier
    ), "Loaded model should be an XGBClassifier"


def test_save_model_none():
    with pytest.raises(ValueError, match="Model cannot be None"):
        save_model(None, "dummy.joblib")


def test_load_model_not_found():
    with pytest.raises(
        FileNotFoundError, match="Model file non_existent.joblib not found"
    ):
        load_model("non_existent.joblib")
