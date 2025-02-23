"""Module for evaluating machine learning models."""

from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from typing import Any

def evaluate_model(model: Any, X_test: Any, y_test: Any, run_id: str = None) -> None:
    """
    Evaluate a model, log metrics to MLflow, and print classification metrics.

    Args:
        model: Trained model (e.g., XGBoost model).
        X_test: Testing features (e.g., pandas DataFrame or numpy array).
        y_test: Testing labels (e.g., pandas Series or numpy array).
        run_id: Optional MLflow run ID to log to an existing run.

    Raises:
        ValueError: If input data or model is invalid.
    """
    if len(X_test) == 0 or len(y_test) == 0 or model is None:
        raise ValueError("Test data, labels, or model cannot be empty or None")

    # Use existing run if run_id is provided and active, otherwise start a new one
    if run_id and mlflow.active_run():
        _evaluate_model(model, X_test, y_test)
    else:
        with mlflow.start_run():
            _evaluate_model(model, X_test, y_test)

def _evaluate_model(model: Any, X_test: Any, y_test: Any) -> None:
    """Helper function to evaluate the model and log metrics."""
    predictions = model.predict(X_test)
    
    print("\n🔹 Evaluating model...")
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
    print("🔹 Evaluation complete")
