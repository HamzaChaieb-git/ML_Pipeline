"""Module for evaluating machine learning models."""

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss
)
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
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate detailed metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba[:, 1]),
        "log_loss": log_loss(y_test, y_pred_proba)
    }
    
    # Calculate class-specific metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Add class-specific metrics
    metrics.update({
        "precision_class_0": report['0']['precision'],
        "recall_class_0": report['0']['recall'],
        "f1_score_class_0": report['0']['f1-score'],
        "precision_class_1": report['1']['precision'],
        "recall_class_1": report['1']['recall'],
        "f1_score_class_1": report['1']['f1-score']
    })
    
    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)

    # Calculate confusion matrix percentages for better interpretation
    confusion_pct = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    
    # Log confusion matrix values
    mlflow.log_metric("true_negative", confusion[0, 0])
    mlflow.log_metric("false_positive", confusion[0, 1])
    mlflow.log_metric("false_negative", confusion[1, 0])
    mlflow.log_metric("true_positive", confusion[1, 1])
    
    # Log confusion matrix percentages
    mlflow.log_metric("true_negative_rate", confusion_pct[0, 0])
    mlflow.log_metric("false_positive_rate", confusion_pct[0, 1])
    mlflow.log_metric("false_negative_rate", confusion_pct[1, 0])
    mlflow.log_metric("true_positive_rate", confusion_pct[1, 1])

    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("🔹 Evaluation complete")
