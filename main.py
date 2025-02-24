"""Enhanced main module for running the ML pipeline with MLflow native tracing."""

import argparse
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from datetime import datetime
from typing import Tuple
import sys
import json

from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model

def setup_enhanced_mlflow():
    """Setup enhanced MLflow tracking with custom configuration."""
    # Set up MLflow tracking URI to match existing configuration
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
            artifact_location=os.path.abspath("./mlruns"),
            tags=experiment_tags
        )
        print(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_system_info(run):
    """Log system and environment information within an MLflow run."""
    import platform
    import psutil

    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "processor": platform.processor(),
        "memory_total": round(psutil.virtual_memory().total / (1024 * 1024 * 1024), 2),  # GB
        "memory_available": round(psutil.virtual_memory().available / (1024 * 1024 * 1024), 2),  # GB
        "cpu_count": psutil.cpu_count(),
        "mlflow_version": mlflow.__version__,
        "tracking_uri": mlflow.get_tracking_uri()
    }

    # Log as parameters
    for key, value in system_info.items():
        if value is not None:
            mlflow.log_param(f"system_{key}", str(value))

    # Save detailed info as JSON artifact
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=4)
    mlflow.log_artifact("system_info.json")
    print("✓ System information logged")

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the enhanced ML pipeline with comprehensive MLflow tracking and native tracing."""
    print("🚀 Launching enhanced ML pipeline...")

    experiment_id = setup_enhanced_mlflow()
    model_version = datetime.now().strftime("%Y%m%d_%H%M")

    with mlflow.start_run(run_name=f"Pipeline_v{model_version}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log run-level metadata for tracing
        mlflow.set_tag("step", "pipeline_execution")
        mlflow.log_param("run_id", run_id)

        # Log system information
        log_system_info(run)

        # Log input parameters
        mlflow.log_params({
            "train_file": train_file,
            "test_file": test_file,
            "model_version": model_version,
            "pipeline_type": "enhanced"
        })

        # Data preparation phase
        mlflow.set_tag("step", "data_preparation")
        print("📊 Preparing data...")
        X_train, X_test, y_train, y_test = process_data(train_file, test_file)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        print("✅ Data preparation complete")

        # Model training phase
        mlflow.set_tag("step", "model_training")
        print("🔧 Training model...")
        model = train_xgb_model(X_train, y_train, model_version=model_version)
        print("✅ Model training complete")

        # Model evaluation phase
        mlflow.set_tag("step", "model_evaluation")
        print("📈 Evaluating model...")
        metrics = evaluate_xgb_model(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(f"metric.{k}", float(v))
        print("✅ Model evaluation complete")

        # Log model
        mlflow.set_tag("step", "model_logging")
        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name=f"churn_model_v{model_version}"
        )
        save_xgb_model(model, f"model_v{model_version}.joblib")

        # Log completion
        mlflow.log_param("completion_time", datetime.now().isoformat())
        mlflow.set_tag("pipeline_status", "completed")
        mlflow.set_tag("step", "completed")

        artifact_uri = run.info.artifact_uri
        print(f"✨ Pipeline completed successfully - Model Version: {model_version}")
        print(f"📁 Artifacts saved to: {artifact_uri}")
        print(f"🔍 MLflow UI: http://localhost:5001")

def main() -> None:
    """Main function to run the enhanced pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced Machine Learning Pipeline")
    parser.add_argument(
        "--train-file",
        type=str,
        default="churn-bigml-80.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="churn-bigml-20.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "evaluate", "all"],
        default="all",
        help="Pipeline action to perform"
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == "all":
            run_enhanced_pipeline(args.train_file, args.test_file)
        elif args.action == "train":
            with mlflow.start_run() as run:
                X_train, _, y_train, _ = process_data(args.train_file, args.test_file)
                train_xgb_model(X_train, y_train)
        elif args.action == "evaluate":
            with mlflow.start_run() as run:
                model = load_xgb_model()
                _, X_test, _, y_test = process_data(args.train_file, args.test_file)
                evaluate_xgb_model(model, X_test, y_test)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
