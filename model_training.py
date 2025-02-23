"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any
from mlflow.models.signature import infer_signature

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
    if (not isinstance(X_train, (np.ndarray, pd.DataFrame)) or 
        not isinstance(y_train, (np.ndarray, pd.Series)) or 
        len(X_train) == 0 or 
        len(y_train) == 0):
        raise ValueError("Training data or labels cannot be empty or invalid")

    # Use existing run if run_id is provided and active, otherwise start a new one
    if run_id and mlflow.active_run():
        model = _train_model(X_train, y_train)
    else:
        with mlflow.start_run():
            model = _train_model(X_train, y_train)
    
    return model

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
        "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle class imbalance
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Log model with signature and input example
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:1] if isinstance(X_train, pd.DataFrame) else X_train[:1]
    mlflow.xgboost.log_model(model, "xgboost_model", signature=signature, input_example=input_example)

    run_id = mlflow.active_run().info.run_id
    print(f"MLflow run ID: {run_id}")
    print("🔹 Training model... Done")
    return model
