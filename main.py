"""Enhanced main module for running the ML pipeline with MLflow native tracing using @mlflow.trace."""

import argparse
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from datetime import datetime
from typing import Tuple, Dict, Any
import sys
import json
import shutil
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model
from mlflow.models.signature import infer_signature

def setup_enhanced_mlflow():
    """Setup enhanced MLflow tracking with custom configuration."""
    # Set up MLflow tracking URI to match Docker setup
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Enable MLflow autologging for relevant frameworks
    mlflow.xgboost.autolog()
    mlflow.sklearn.autolog()

    # Create or get the experiment with custom metadata
    experiment_name = "churn_prediction"
    experiment_tags = {
        "project_name": "churn_prediction",
        "project_version": "enhanced_v1",
        "department": "data_science",
        "owner": "mlops_team",
        "framework": "xgboost",
        "pipeline_type": "binary_classification"
    }

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.abspath("./artifacts/mlruns"),
            tags=experiment_tags
        )
        print(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id

@mlflow.trace
def prepare_data_traced(train_file: str, test_file: str) -> Tuple:
    """Traced data preparation step."""
    X_train, X_test, y_train, y_test = process_data(train_file, test_file)
    
    # Convert integer columns to float64 to handle potential missing values
    integer_columns = ['Customer service calls', 'Total intl calls', 'Number vmail messages']  # Adjust based on your dataset
    for col in integer_columns:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(float)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(float)

    return X_train, X_test, y_train, y_test

@mlflow.trace(attributes={"step": "model_training"})
def train_model_traced(X_train: Any, y_train: Any, model_version: str) -> Any:
    """Traced model training step."""
    return train_xgb_model(X_train, y_train, model_version=model_version)

@mlflow.trace(attributes={"step": "model_evaluation"})
def evaluate_model_traced(model: Any, X_test: Any, y
