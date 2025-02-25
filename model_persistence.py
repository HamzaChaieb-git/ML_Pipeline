"""Module for saving and loading machine learning models."""

import joblib
import os
import mlflow
from typing import Any


def save_model(model: Any, filename: str = "model.joblib") -> None:
    """
    Save a model using joblib and log it with MLflow.

    Args:
        model: Trained model to save.
        filename (str): Name of the file to save the model to (default: "model.joblib").

    Raises:
        ValueError: If the model is None.
        IOError: If there's an issue saving the model file.
    """
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")
        # Log artifact in the current run without creating a new one
        mlflow.log_artifact(filename)
    except Exception as e:
        raise IOError(f"Failed to save model to {filename}: {e}")


def load_model(filename: str = "model.joblib") -> Any:
    """
    Load a model using joblib.

    Args:
        filename (str): Name of the file to load the model from (default: "model.joblib").

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the model file is not found.
        IOError: If there's an issue loading the model file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found")

    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from {filename}: {e}")
