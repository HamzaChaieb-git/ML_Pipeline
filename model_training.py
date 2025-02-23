"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd  # Added this import
import numpy as np
from typing import Any


def train_model(X_train: Any, y_train: Any, run_id: str = None) -> xgb.XGBClassifier:
    """
    Train an XGBoost model and log it with MLflow for tracking.

    Args:
        X_train: Training features (e.g., pandas DataFrame or numpy array).
        y_train: Training labels (e.g., pandas Series or numpy array).
        run_id: Optional MLflow run ID to log to an existing run.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.

    Raises:
        ValueError: If input data is invalid or empty.
    """
    # Check if X_train or y_train is empty (works for NumPy arrays or pandas objects)
    if (not isinstance(X_train, (np.ndarray, pd.DataFrame)) or 
        not isinstance(y_train, (np.ndarray, pd.Series)) or 
        len(X_train) == 0 or 
        len(y_train) == 0):
        raise ValueError("Training data or labels cannot be empty or invalid")

    # Use existing run if run_id is provided, otherwise start a new one
    if run_id:
        with mlflow.start_run(run_id=run_id):
            return _train_model(X_train, y_train)
    else:
        with mlflow.start_run():
            return _train_model(X_train, y_train)


def _train_model(X_train: Any, y_train: Any) -> xgb.XGBClassifier:
    """Helper function to train the model and log parameters."""
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": 42,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    mlflow.xgboost.log_model(model, "xgboost_model")
    return model
