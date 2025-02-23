"""Module for evaluating machine learning models."""

from sklearn.metrics import classification_report, confusion_matrix
from typing import Any

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> None:
    """
    Evaluate a model and print classification metrics.

    Args:
        model: Trained model (e.g., XGBoost model).
        X_test: Testing features (e.g., pandas DataFrame or numpy array).
        y_test: Testing labels (e.g., pandas Series or numpy array).

    Raises:
        ValueError: If input data or model is invalid.
    """
    if len(X_test) == 0 or len(y_test) == 0 or model is None:
        raise ValueError("Test data, labels, or model cannot be empty or None")

    _evaluate_model(model, X_test, y_test)

def _evaluate_model(model: Any, X_test: Any, y_test: Any) -> None:
    """Helper function to evaluate the model and print metrics."""
    predictions = model.predict(X_test)
    
    print("\n🔹 Evaluating model...")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("🔹 Evaluation complete")
