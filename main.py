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
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    mlflow.xgboost.autolog()
    mlflow.sklearn.autolog()

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
    
    integer_columns = ['Customer service calls', 'Total intl calls', 'Number vmail messages']
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
def evaluate_model_traced(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Traced model evaluation step."""
    return evaluate_xgb_model(model, X_test, y_test)

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline with specified action")
    parser.add_argument("--train-file", required=True, help="Path to training data CSV")
    parser.add_argument("--test-file", required=True, help="Path to testing data CSV")
    parser.add_argument("--action", choices=["train", "evaluate", "all"], default="all",
                        help="Action to perform: 'train', 'evaluate', or 'all'")
    args = parser.parse_args()

    # Set up MLflow
    experiment_id = setup_enhanced_mlflow()

    # Generate model version based on timestamp
    model_version = datetime.now().strftime("v%Y%m%d_%H%M")

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data_traced(args.train_file, args.test_file)

        model = None
        if args.action in ["train", "all"]:
            # Train model
            model = train_model_traced(X_train, y_train, model_version)
            model_filename = os.path.join("artifacts", "models", f"model_{model_version}.joblib")
            save_xgb_model(model, model_filename)

        if args.action in ["evaluate", "all"]:
            # Load model if not trained in this run
            if model is None:
                latest_model = max(
                    [f for f in os.listdir(os.path.join("artifacts", "models")) if f.startswith("model_")],
                    key=lambda x: os.path.getctime(os.path.join("artifacts", "models", x)),
                    default=None
                )
                if latest_model:
                    model = load_xgb_model(os.path.join("artifacts", "models", latest_model))
                else:
                    raise FileNotFoundError("No model found for evaluation")

            # Evaluate model
            metrics = evaluate_model_traced(model, X_test, y_test)
            print(f"Evaluation completed with metrics: {metrics}")

if __name__ == "__main__":
    main()