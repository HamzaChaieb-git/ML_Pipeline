"""Module for evaluating machine learning models."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
    if len(X_test) == 0 or len(y_test) == 0 or model is None:
        raise ValueError("Test data, labels, or model cannot be empty or None")

    print("\n🔹 Evaluating model...")
    
    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, predictions, output_dict=True)
    confusion = confusion_matrix(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    # Log metrics to MLflow
    metrics = {
        "accuracy": report['accuracy'],
        "precision_0": report['0']['precision'],
        "recall_0": report['0']['recall'],
        "f1_0": report['0']['f1-score'],
        "precision_1": report['1']['precision'],
        "recall_1": report['1']['recall'],
        "f1_1": report['1']['f1-score'],
        "roc_auc": roc_auc
    }
    mlflow.log_metrics(metrics)

    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print("🔹 Evaluation complete")
