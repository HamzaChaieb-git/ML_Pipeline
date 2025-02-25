import pytest
import os
from model_persistence import save_model, load_model
import xgboost as xgb
import mlflow

# Fixture for sample model and temporary filename
@pytest.fixture
def sample_model(tmp_path):
    model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False)
    filename = tmp_path / "test_model.joblib"
    return model, str(filename)

def test_save_model(sample_model, monkeypatch):
    # Mock mlflow to prevent actual logging during tests
    monkeypatch.setattr(mlflow, "log_artifact", lambda path, run_id=None: None)
    
    model, filename = sample_model
    save_model(model, filename)
    assert os.path.exists(filename), "Model file should be saved"

def test_load_model(sample_model):
    model, filename = sample_model
    save_model(model, filename)
    loaded_model = load_model(filename)
    assert isinstance(loaded_model, xgb.XGBClassifier), "Loaded model should be an XGBClassifier"

def test_save_model_none(monkeypatch):
    # Mock mlflow to prevent actual logging during tests
    monkeypatch.setattr(mlflow, "log_artifact", lambda path, run_id=None: None)
    
    with pytest.raises(ValueError, match="Model cannot be None"):
        save_model(None, "dummy.joblib")

def test_load_model_not_found():
    with pytest.raises(FileNotFoundError, match="Model file non_existent.joblib not found"):
        load_model("non_existent.joblib")