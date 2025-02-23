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
import pandas as pd
from typing import Any, Dict

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a model, log metrics to MLflow, and print classification metrics.

    Args:
        model: Trained model (e.g., XGBoost model).
        X_test: Testing features (e.g., pandas DataFrame or numpy array).
        y_test: Testing labels (e.g., pandas Series or numpy array).

    Returns:
        Dictionary containing all computed metrics.

    Raises:
        ValueError: If input data or model is invalid.
    """
    if len(X_test) == 0 or len(y_test) == 0 or model is None:
        raise ValueError("Test data, labels, or model cannot be empty or None")

    print("\n🔹 Evaluating model...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
        "test_log_loss": log_loss(y_test, y_pred_proba)
    }
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Add detailed metrics
    metrics.update({
        "test_precision_class_0": report['0']['precision'],
        "test_recall_class_0": report['0']['recall'],
        "test_f1_class_0": report['0']['f1-score'],
        "test_precision_class_1": report['1']['precision'],
        "test_recall_class_1": report['1']['recall'],
        "test_f1_class_1": report['1']['f1-score']
    })
    
    # Calculate and add confusion matrix metrics
    metrics.update({
        "test_true_negatives": int(confusion[0, 0]),
        "test_false_positives": int(confusion[0, 1]),
        "test_false_negatives": int(confusion[1, 0]),
        "test_true_positives": int(confusion[1, 1])
    })
    
    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)
    
    # Create confusion matrix figure
    confusion_df = pd.DataFrame(
        confusion,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    
    # Save confusion matrix as CSV artifact
    confusion_df.to_csv("confusion_matrix.csv")
    mlflow.log_artifact("confusion_matrix.csv")

    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("🔹 Evaluation complete")
    
    return metrics
