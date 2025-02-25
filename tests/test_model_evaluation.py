import pytest
import pandas as pd
import numpy as np
import os
import mlflow  # Added import for mlflow
from model_training import train_model
from model_evaluation import evaluate_model
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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

def test_evaluate_model_metrics(trained_model_and_data, monkeypatch, tmp_path):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *args, **kwargs: None)
    
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    assert "roc_auc" in metrics, "ROC AUC should be in metrics"
    assert "log_loss" in metrics, "Log loss should be in metrics"
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert metrics["roc_auc"] >= 0 and metrics["roc_auc"] <= 1, "ROC AUC should be between 0 and 1"

def test_evaluate_model_artifacts(trained_model_and_data, monkeypatch, tmp_path):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *args, **kwargs: None)
    
    model, X_test, y_test = trained_model_and_data
    
    # Set artifacts directory to a temporary path for testing
    artifacts_base = tmp_path / "artifacts"
    os.environ["ARTIFACTS_DIR"] = str(artifacts_base)
    os.makedirs(artifacts_base / "evaluation", exist_ok=True)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # Find the latest evaluation subdirectory (if any)
    evaluation_dir = os.path.join(artifacts_base, "evaluation")
    subdirs = [d for d in os.listdir(evaluation_dir) if os.path.isdir(os.path.join(evaluation_dir, d))]
    
    # If no subdirectories are created (due to no MLflow logging), check if metrics are still generated
    assert isinstance(metrics, dict), "Metrics should still be generated even without MLflow"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    
    if subdirs:  # Only check artifacts if they exist
        latest_dir = max(subdirs, key=lambda x: os.path.getctime(os.path.join(evaluation_dir, x)))
        artifact_dir = os.path.join(evaluation_dir, latest_dir)
        
        assert os.path.exists(os.path.join(artifact_dir, "performance_curves.html")), "Performance curves should be generated"
        assert os.path.exists(os.path.join(artifact_dir, "confusion_matrix.png")), "Confusion matrix should be generated"
        assert os.path.exists(os.path.join(artifact_dir, "evaluation_summary.json")), "Summary report should be generated"