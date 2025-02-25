import pytest
import pandas as pd
import numpy as np
import os
import mlflow  # Added import for mlflow
from main import prepare_data_traced, train_model_traced, evaluate_model_traced, setup_enhanced_mlflow
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb  # Import for type checking

# Fixture for sample data files
@pytest.fixture
def sample_files(tmp_path):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    data = {
        "Total day minutes": [120.5, 150.3, 130.0, 140.0],
        "Customer service calls": [3, 2, 4, 1],
        "International plan": ["no", "yes", "no", "yes"],
        "Total intl minutes": [10.2, 8.5, 12.0, 9.5],
        "Total intl calls": [5, 4, 6, 3],
        "Total eve minutes": [200.0, 180.5, 190.0, 210.0],
        "Number vmail messages": [0, 5, 0, 3],
        "Voice mail plan": ["no", "yes", "no", "yes"],
        "Churn": [0, 1, 0, 1]  # Ensure both classes (0 and 1) are present
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:2]  # First two rows for training
    test_df = df.iloc[2:]   # Last two rows for testing
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    return str(train_file), str(test_file)

def test_prepare_data_traced(sample_files, monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda *args, **kwargs: None)
    
    train_file, test_file = sample_files
    X_train, X_test, y_train, y_test = prepare_data_traced(train_file, test_file)
    
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array"

def test_train_model_traced(sample_files, monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow.xgboost, "log_model", lambda *args, **kwargs: None)
    
    train_file, _ = sample_files
    df = pd.read_csv(train_file)
    X_train = df.drop("Churn", axis=1)
    
    # Encode categorical columns before training
    le_international = LabelEncoder()
    le_voice = LabelEncoder()
    X_train["International plan"] = le_international.fit_transform(X_train["International plan"])
    X_train["Voice mail plan"] = le_voice.fit_transform(X_train["Voice mail plan"])
    
    y_train = df["Churn"].values
    
    model = train_model_traced(X_train, y_train, model_version="test_1.0")
    assert isinstance(model, xgb.XGBClassifier), "Model should be an XGBClassifier"

def test_evaluate_model_traced(sample_files, monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *args, **kwargs: None)
    
    train_file, test_file = sample_files
    X_train, X_test, y_train, y_test = prepare_data_traced(train_file, test_file)
    
    # Encode categorical columns in X_test
    le_international = LabelEncoder()
    le_voice = LabelEncoder()
    X_test["International plan"] = le_international.fit_transform(X_test["International plan"])
    X_test["Voice mail plan"] = le_voice.fit_transform(X_test["Voice mail plan"])
    
    model = train_model_traced(X_train, y_train, model_version="test_1.0")
    metrics = evaluate_model_traced(model, X_test, y_test)
    
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Accuracy should be in metrics"

def test_setup_enhanced_mlflow(monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "create_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow.xgboost, "autolog", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow.sklearn, "autolog", lambda *args, **kwargs: None)
    
    # Since MLflow is disabled, this test can return a dummy value or skip MLflow-specific assertions
    result = setup_enhanced_mlflow()
    assert result is None or isinstance(result, str), "Should return None or a string (e.g., experiment ID)"