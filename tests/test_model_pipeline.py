import pytest
import pandas as pd
import numpy as np
import mlflow  # Added import for mlflow
import xgboost as xgb
from data_processing import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model
from sklearn.preprocessing import LabelEncoder

# Fixture for sample data (mimics your churn CSV structure)
@pytest.fixture
def sample_data(tmp_path):
    train_file = tmp_path / "churn-bigml-80.csv"
    test_file = tmp_path / "churn-bigml-20.csv"
    
    data = {
        "Total day minutes": [120.5, 150.3, 130.0, 140.0],
        "Customer service calls": [3, 2, 4, 1],
        "International plan": ["no", "yes", "no", "yes"],
        "Total intl minutes": [10.2, 8.5, 12.0, 9.5],
        "Total intl calls": [5, 4, 6, 3],
        "Total eve minutes": [200.0, 180.5, 190.0, 210.0],
        "Number vmail messages": [0, 5, 0, 3],
        "Voice mail plan": ["no", "yes", "no", "yes"],
        "Churn": [0, 1, 0, 1]  # Ensure both classes are present
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:2]  # First two rows for training
    test_df = df.iloc[2:]   # Last two rows for testing
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    return str(train_file), str(test_file)

@pytest.fixture
def sample_train_data():
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
    
    return X_train, y_train

def test_prepare_data_output_shapes(sample_data, monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda *args, **kwargs: None)
    
    train_file, test_file = sample_data
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
    
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array"
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have the same number of rows"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have the same number of rows"
    assert len(X_train.columns) == 8, "X_train should have 8 features"

def test_train_model_returns_model(sample_train_data, monkeypatch):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow.xgboost, "log_model", lambda *args, **kwargs: None)
    
    X_train, y_train = sample_train_data
    model = train_model(X_train, y_train, model_version="test_1.0")
    assert isinstance(model, xgb.XGBClassifier), "Model should be an XGBClassifier"
    assert model is not None, "Model should not be None"

def test_evaluate_model_metrics(sample_train_data, monkeypatch, tmp_path):
    # Disable MLflow logging completely
    monkeypatch.setattr(mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *args, **kwargs: None)
    
    X_train, y_train = sample_train_data
    model = train_model(X_train, y_train, model_version="test_1.0")
    metrics = evaluate_model(model, X_train, y_train)
    
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    assert "roc_auc" in metrics, "ROC AUC should be in metrics"
    assert "log_loss" in metrics, "Log loss should be in metrics"
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert metrics["roc_auc"] >= 0 and metrics["roc_auc"] <= 1, "ROC AUC should be between 0 and 1"