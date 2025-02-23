"""Module for training machine learning models using XGBoost."""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Any

def train_model(X_train: Any, y_train: Any) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.

    Args:
        X_train: Training features (e.g., pandas DataFrame or numpy array).
        y_train: Training labels (e.g., pandas Series or numpy array).

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

    model = _train_model(X_train, y_train)
    return model

def _train_model(X_train: Any, y_train: Any) -> xgb.XGBClassifier:
    """Helper function to train the model."""
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

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("🔹 Training model... Done")
    return model
