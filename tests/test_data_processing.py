import pytest
import pandas as pd
import numpy as np
from data_processing import prepare_data

# Fixture for sample data (mimics your churn CSV structure)
@pytest.fixture
def sample_data(tmp_path):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    # Sample data matching expected features from app.py
    data = {
        "Total day minutes": [120.5, 150.3],
        "Customer service calls": [3, 2],
        "International plan": ["no", "yes"],
        "Total intl minutes": [10.2, 8.5],
        "Total intl calls": [5, 4],
        "Total eve minutes": [200.0, 180.5],
        "Number vmail messages": [0, 5],
        "Voice mail plan": ["no", "yes"],
        "Churn": [0, 1]  # Target variable
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:1]
    test_df = df.iloc[1:]
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    return str(train_file), str(test_file)

def test_prepare_data_output_shapes(sample_data):
    train_file, test_file = sample_data
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
    
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array"
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have same number of rows"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have same number of rows"

def test_prepare_data_feature_names(sample_data):
    train_file, test_file = sample_data
    X_train, X_test, _, _ = prepare_data(train_file, test_file)
    
    expected_features = [
        "Total day minutes", "Customer service calls", "International plan",
        "Total intl minutes", "Total intl calls", "Total eve minutes",
        "Number vmail messages", "Voice mail plan"
    ]
    assert all(f in X_train.columns for f in expected_features), "X_train missing expected features"
    assert all(f in X_test.columns for f in expected_features), "X_test missing expected features"
