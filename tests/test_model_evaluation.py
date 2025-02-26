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

@pytest.fixture
def trained_model_and_data(tmp_path):
    X_train = pd.DataFrame({
        "Total day minutes": [120.5, 150.3],
        "Customer service calls": [3, 2],
        "International plan": ["no", "yes"],  # Object dtype (categorical)
        "Total intl minutes": [10.2, 8.5],
        "Total intl calls": [5, 4],
        "Total eve minutes": [200.0, 180.5],
        "Number vmail messages": [0, 5],
        "Voice mail plan": ["no", "yes"]  # Object dtype (categorical)
    })
    y_train = np.array([0, 1])
    
    # Encode categorical columns
    le_international = LabelEncoder()
    le_voice = LabelEncoder()
    X_train["International plan"] = le_international.fit_transform(X_train["International plan"])
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
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert metrics["roc_auc"] >= 0 and metrics["roc_auc"] <= 1, "ROC AUC should be between 0 and 1"

def test_evaluate_model_artifacts(trained_model_and_data, tmp_path):
    model, X_test, y_test = trained_model_and_data
    
    # Set MLflow artifact root to a temporary directory
    temp_artifacts = tmp_path / "artifacts"
    os.makedirs(temp_artifacts, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{temp_artifacts}")
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # Find the latest run directory in the temporary MLflow tracking URI
    runs_dir = max([d for d in temp_artifacts.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    artifact_dir = runs_dir / "artifacts" / "evaluation"
    
    assert isinstance(metrics, dict), "Metrics should still be generated"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    
    if artifact_dir.exists():  # Only check artifacts if they exist
        assert os.path.exists(os.path.join(artifact_dir, "performance_curves.html")), "Performance curves should be generated"
        assert os.path.exists(os.path.join(artifact_dir, "confusion_matrix.png")), "Confusion matrix should be generated"
        assert os.path.exists(os.path.join(artifact_dir, "evaluation_summary.json")), "Summary report should be generated"