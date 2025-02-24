"""Enhanced module for evaluating machine learning models with comprehensive MLflow tracking."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    calinski_harabasz_score
)
from sklearn.calibration import calibration_curve
import mlflow
from typing import Any, Dict, List, Tuple
import json
from datetime import datetime

def create_evaluation_artifacts(y_test: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray, feature_names: List[str]) -> None:
    """Create and log comprehensive evaluation artifacts."""
    
    # Create artifacts directory
    artifact_dir = f"evaluation_artifacts_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. ROC and PR Curves (Interactive)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC Curve', 'Precision-Recall Curve'))
    
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name='ROC curve'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=recall, y=precision, name=f'PR curve (AP={avg_precision:.2f})'),
        row=1, col=2
    )
    
    fig.update_layout(height=600, width=1200, title_text="Model Performance Curves")
    fig.write_html(f"{artifact_dir}/performance_curves.html")
    
    # 2. Prediction Distribution
    fig = px.histogram(
        y_pred_proba[:, 1], 
        nbins=50,
        title="Prediction Probability Distribution",
        labels={'value': 'Probability', 'count': 'Frequency'}
    )
    fig.write_html(f"{artifact_dir}/prediction_distribution.html")
    
    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(f"{artifact_dir}/confusion_matrix.png")
    plt.close()
    
    # 4. Calibration Plot
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibration'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(dash='dash')))
    fig.update_layout(title='Model Calibration Plot', xaxis_title='Mean Predicted Probability', yaxis_title='True Probability')
    fig.write_html(f"{artifact_dir}/calibration_plot.html")
    
    # Log all artifacts
    mlflow.log_artifacts(artifact_dir)

def create_summary_report(metrics: Dict[str, float], artifact_dir: str) -> None:
    """Create and save a comprehensive summary report."""
    report = {
        "model_performance": {
            "overall_metrics": {
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"]
            },
            "class_metrics": {
                "class_0": {
                    "precision": metrics["precision_class_0"],
                    "recall": metrics["recall_class_0"],
                    "f1": metrics["f1_class_0"]
                },
                "class_1": {
                    "precision": metrics["precision_class_1"],
                    "recall": metrics["recall_class_1"],
                    "f1": metrics["f1_class_1"]
                }
            },
            "confusion_matrix": {
                "true_negatives": metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "true_positives": metrics["true_positives"]
            },
            "derived_metrics": {
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision_recall_ratio": metrics["precision_recall_ratio"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"]
            }
        },
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_recommendation": "production" if metrics["accuracy"] > 0.85 and metrics["roc_auc"] > 0.85 else "staging"
    }
    
    with open(f"{artifact_dir}/evaluation_summary.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(f"{artifact_dir}/evaluation_summary.json")

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Enhanced model evaluation with comprehensive metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Create artifacts directory
    artifact_dir = f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(artifact_dir, exist_ok=True)

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
    
    # Add advanced metrics
    metrics.update({
        "balanced_accuracy": (metrics["recall_class_0"] + metrics["recall_class_1"]) / 2,
        "precision_recall_ratio": metrics["precision"] / metrics["recall"] if metrics["recall"] > 0 else 0,
        "false_positive_rate": metrics["false_positives"] / (metrics["false_positives"] + metrics["true_negatives"]),
        "false_negative_rate": metrics["false_negatives"] / (metrics["false_negatives"] + metrics["true_positives"]),
        "positive_predictive_value": metrics["precision"],
        "negative_predictive_value": metrics["true_negatives"] / (metrics["true_negatives"] + metrics["false_negatives"])
    })
    
    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)
    
    # Create and log evaluation artifacts
    if isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    create_evaluation_artifacts(y_test, y_pred, y_pred_proba, feature_names)
    
    # Create and log summary report
    create_summary_report(metrics, artifact_dir)
    
    # Print evaluation results
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nKey Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics
