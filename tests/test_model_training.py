import pytest
import pandas as pd
import numpy as np
from model_training import train_model
import xgboost as xgb

# Fixture for sample training data
@pytest.fixture
def sample_train_data():
    X_train = pd.DataFrame({
        "Total day minutes": [120.5, 150.3],
        "Customer service calls": [3, 2],
        "International plan": [0, 1],
        "Total intl minutes": [10.2, 8.5],
        "Total intl calls": [5, 4],
        "Total eve minutes": [200.0, 180.5],
        "Number vmail messages": [0, 5],
        "Voice mail plan": [0, 1]
    })
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
