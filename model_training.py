"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
from typing import Any


def train_model(X_train: Any, y_train: Any) -> xgb.XGBClassifier:
    """
    Train an XGBoost model and log it with MLflow for tracking.

    Args:
        X_train: Training features (e.g., pandas DataFrame or numpy array).
        y_train: Training labels (e.g., pandas Series or numpy array).

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.

    Raises:
        ValueError: If input data is invalid or empty.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data or labels cannot be empty")

    with mlflow.start_run():
        params = {
            "objective": "binary:logistic",  # Binary classification
            "max_depth": 6,                 # Maximum depth of trees
            "learning_rate": 0.1,           # Step size shrinkage
            "n_estimators": 100,            # Number of boosting rounds
            "random_state": 42,             # For reproducibility
            "min_child_weight": 1,          # Minimum sum of instance weight needed in a child
            "subsample": 0.8,               # Subsample ratio of the training instance
            "colsample_bytree": 0.8,        # Subsample ratio of columns when constructing each tree
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        mlflow.xgboost.log_model(model, "xgboost_model")
        
        return model
