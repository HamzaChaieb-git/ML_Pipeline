"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any, Union
from sklearn.metrics import log_loss

def train_model(X_train: Union[pd.DataFrame, Any], y_train: Any) -> xgb.XGBClassifier:
    """
    Train an XGBoost model and log it with MLflow for tracking.

    Args:
        X_train: Training features (pandas DataFrame or numpy array)
        y_train: Training labels (pandas Series or numpy array)

    Returns:
        xgb.XGBClassifier: Trained XGBoost model

    Raises:
        ValueError: If input data is invalid or empty
    """
    if isinstance(X_train, pd.DataFrame) and X_train.empty or y_train is None or len(y_train) == 0:
        raise ValueError("Training data or labels cannot be empty")

    # Define model parameters
    params = {
        "objective": "binary:logistic",  # Binary classification
        "max_depth": 6,                  # Maximum depth of trees
        "learning_rate": 0.1,            # Step size shrinkage
        "n_estimators": 100,             # Number of boosting rounds
        "random_state": 42,              # For reproducibility
        "min_child_weight": 1,           # Minimum sum of instance weight needed in a child
        "subsample": 0.8,                # Subsample ratio of the training instance
        "colsample_bytree": 0.8,         # Subsample ratio of columns when constructing each tree
        "enable_categorical": True,       # Enable categorical feature support
        "tree_method": "hist",           # Use histogram-based algorithm for faster training
    }
    
    # Log parameters to MLflow
    mlflow.log_params(params)

    # Initialize and train the model
    model = xgb.XGBClassifier(**params)
    
    # Convert data types for categorical columns if they exist
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category')

    # Fit the model
    model.fit(X_train, y_train, verbose=True)

    # Log training metrics
    y_pred_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred_proba)
    mlflow.log_metric("train_logloss", train_loss)

    # Log the model to MLflow
    input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5]
    mlflow.xgboost.log_model(
        model, 
        "xgboost_model",
        input_example=input_example
    )
    
    # Log feature importance
    if isinstance(X_train, pd.DataFrame):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log feature importance as a JSON file
        feature_importance.to_json("feature_importance.json")
        mlflow.log_artifact("feature_importance.json")
        
        # Log feature importance scores
        for feature, importance in zip(feature_importance['feature'], feature_importance['importance']):
            mlflow.log_metric(f"feature_importance_{feature}", importance)
    
    return model
