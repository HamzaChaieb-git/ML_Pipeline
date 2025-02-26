import pytest
import pandas as pd
import numpy as np
import os
from data_processing import prepare_data
from sklearn.preprocessing import LabelEncoder


# Fixture for sample data (mimics your churn CSV structure)
@pytest.fixture
def sample_data(tmp_path):
    train_file = tmp_path / "churn-bigml-80.csv"
    test_file = tmp_path / "churn-bigml-20.csv"

    # Sample data matching expected features from data_processing.py
    data = {
        "Total day minutes": [120.5, 150.3, 130.0, 140.0],
        "Customer service calls": [3, 2, 4, 1],
        "International plan": ["no", "yes", "no", "yes"],
        "Total intl minutes": [10.2, 8.5, 12.0, 9.5],
        "Total intl calls": [5, 4, 6, 3],
        "Total eve minutes": [200.0, 180.5, 190.0, 210.0],
        "Number vmail messages": [0, 5, 0, 3],
        "Voice mail plan": ["no", "yes", "no", "yes"],
        "Churn": [0, 1, 0, 1],  # Ensure both classes are present
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:2]  # First two rows for training
    test_df = df.iloc[2:]  # Last two rows for testing

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
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "X_train and y_train should have the same number of rows"
    assert (
        X_test.shape[0] == y_test.shape[0]
    ), "X_test and y_test should have the same number of rows"
    assert len(X_train.columns) == 8, "X_train should have 8 features"


def test_prepare_data_categorical_encoding(sample_data):
    train_file, test_file = sample_data
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)

    categorical_features = ["International plan", "Voice mail plan"]
    for feature in categorical_features:
        assert (
            X_train[feature].dtype == int
        ), f"{feature} in X_train should be encoded as integers"
        assert (
            X_test[feature].dtype == int
        ), f"{feature} in X_test should be encoded as integers"
        assert (
            X_train[feature].min() >= 0
        ), f"{feature} in X_train should have non-negative values"
        assert (
            X_test[feature].min() >= 0
        ), f"{feature} in X_test should have non-negative values"


def test_prepare_data_missing_files():
    with pytest.raises(FileNotFoundError, match="Could not find data files:"):
        prepare_data("nonexistent_train.csv", "nonexistent_test.csv")


def test_prepare_data_missing_columns(sample_data):
    train_file, _ = sample_data
    df = pd.read_csv(train_file)
    df.drop("Churn", axis=1, inplace=True)  # Remove required column
    df.to_csv(train_file, index=False)

    with pytest.raises(ValueError, match="Missing columns in training data:"):
        prepare_data(train_file, train_file)
