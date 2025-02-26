import pytest
import os
import mlflow
import tempfile
from model_persistence import save_model, load_model
import xgboost as xgb
import joblib

# Fixture to set up MLflow tracking URI in a temporary directory
@pytest.fixture(autouse=True)
def setup_mlflow_tracking(tmp_path):
    # Set MLflow tracking URI to a temporary directory
    temp_dir = tmp_path / "mlruns"
    os.makedirs(temp_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    yield
    # Clean up (optional, but can help avoid permission issues in future runs)
    # shutil.rmtree(temp_dir, ignore_errors=True)

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
    assert isinstance(loaded_model, xgb.XGBClassifier), "Loaded model should be an XGBClassifier"

def test_save_model_none():
    with pytest.raises(ValueError, match="Model cannot be None"):
        save_model(None, "dummy.joblib")

def test_load_model_not_found():
    with pytest.raises(FileNotFoundError, match="Model file non_existent.joblib not found"):
        load_model("non_existent.joblib")