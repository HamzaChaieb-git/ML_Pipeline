import pytest
import os
from main import prepare_data_traced, train_model_traced, evaluate_model_traced
from unittest.mock import patch, Mock

@pytest.fixture
def sample_files(tmp_path):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    data = {
        "Total day minutes": [120.5, 150.3],
        "Customer service calls": [3, 2],
        "International plan": ["no", "yes"],
        "Total intl minutes": [10.2, 8.5],
        "Total intl calls": [5, 4],
        "Total eve minutes": [200.0, 180.5],
        "Number vmail messages": [0, 5],
        "Voice mail plan": ["no", "yes"],
        "Churn": [0, 1]
    }
    df = pd.DataFrame(data)
    df.iloc[:1].to_csv(train_file, index=False)
    df.iloc[1:].to_csv(test_file, index=False)
    return str(train_file), str(test_file)

def test_prepare_data_traced(sample_files):
    train_file, test_file = sample_files
    X_train, X_test, y_train, y_test = prepare_data_traced(train_file, test_file)
    assert X_train.shape[0] > 0, "X_train should have data"
    assert X_test.shape[0] > 0, "X_test should have data"

def test_train_model_traced(sample_files):
    train_file, test_file = sample_files
    X_train, _, y_train, _ = prepare_data_traced(train_file, test_file)
    model = train_model_traced(X_train, y_train, model_version="test_main_1.0")
    assert model is not None, "Trained model should not be None"

def test_evaluate_model_traced(sample_files):
    train_file, test_file = sample_files
    X_train, X_test, y_train, y_test = prepare_data_traced(train_file, test_file)
    model = train_model_traced(X_train, y_train, model_version="test_main_1.0")
    metrics = evaluate_model_traced(model, X_test, y_test)
    assert "accuracy" in metrics, "Evaluation should return accuracy"
