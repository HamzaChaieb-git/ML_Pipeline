import pytest
import pandas as pd
import numpy as np
from model_evaluation import evaluate_model
import xgboost as xgb

@pytest.fixture
def trained_model_and_data():
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
    model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False)
    model.fit(X_train, y_train)
    return model, X_train, y_train

def test_evaluate_model_metrics(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "accuracy" in metrics, "Accuracy should be in metrics"
    assert "roc_auc" in metrics, "ROC AUC should be in metrics"
    assert all(isinstance(v, (int, float)) for v in metrics.values()), "Metrics should be numeric"

def test_evaluate_model_artifacts(trained_model_and_data):
    import os
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    
    artifact_dir = max([d for d in os.listdir("artifacts/evaluation") if os.path.isdir(f"artifacts/evaluation/{d}")], 
                      key=lambda x: os.path.getctime(f"artifacts/evaluation/{x}"))
    assert os.path.exists(f"artifacts/evaluation/{artifact_dir}/performance_curves.html"), "Performance curves should be generated"
    assert os.path.exists(f"artifacts/evaluation/{artifact_dir}/confusion_matrix.png"), "Confusion matrix should be generated"
