"""Module for evaluating machine learning models."""

from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from typing import Any


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> None:
    """
    Evaluate a model, log metrics to MLflow, and print classification metrics.

    Args:
        model: Trained model (e.g., XGBoost model).
        X_test: Testing features (e.g., pandas DataFrame or numpy array).
        y_test: Testing labels (e.g., pandas Series or numpy array).

    Raises:
        ValueError: If input data or model is invalid.
    """
    if X_test.empty or y_test.empty or model is None:
        raise ValueError("Test data, labels, or model cannot be empty or None")

    with mlflow.start_run():
        predictions = model.predict(X_test)
        
        report = classification_report(y_test, predictions, output_dict=True)
        confusion = confusion_matrix(y_test, predictions)
        
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "precision_0": report['0']['precision'],
            "recall_0": report['0']['recall'],
            "f1_0": report['0']['f1-score'],
            "precision_1": report['1']['precision'],
            "recall_1": report['1']['recall'],
            "f1_1": report['1']['f1-score'],
        })

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion)
