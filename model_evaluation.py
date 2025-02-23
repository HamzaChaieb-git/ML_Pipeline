"""Module for evaluating machine learning models."""

import numpy as np
import pandas as pd
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
from typing import Any, Dict

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a model and log metrics to MLflow.

    Args:
        model: Trained model (e.g., XGBoost model).
        X_test: Testing features (e.g., pandas DataFrame or numpy array).
        y_test: Testing labels (e.g., pandas Series or numpy array).

    Returns:
        Dictionary containing all computed metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
        "log_loss": log_loss(y_test, y_pred_proba)
    }
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Add class-specific metrics
    metrics.update({
        "precision_class_0": report['0']['precision'],
        "recall_class_0": report['0']['recall'],
        "f1_class_0": report['0']['f1-score'],
        "precision_class_1": report['1']['precision'],
        "recall_class_1": report['1']['recall'],
        "f1_class_1": report['1']['f1-score']
    })
    
    # Add confusion matrix metrics
    metrics.update({
        "true_negatives": int(confusion[0, 0]),
        "false_positives": int(confusion[0, 1]),
        "false_negatives": int(confusion[1, 0]),
        "true_positives": int(confusion[1, 1])
    })
    
    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)
    
    # Save confusion matrix
    confusion_df = pd.DataFrame(
        confusion,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    confusion_df.to_csv("confusion_matrix.csv")
    
    # Print evaluation results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics
