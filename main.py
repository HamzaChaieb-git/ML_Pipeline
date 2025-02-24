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
    return X_train, X_test, y_train, y_test

@mlflow.trace(attributes={"step": "model_training"})
def train_model_traced(X_train: Any, y_train: Any, model_version: str) -> Any:
    """Traced model training step."""
    return train_xgb_model(X_train, y_train, model_version=model_version)

@mlflow.trace(attributes={"step": "model_evaluation"})
def evaluate_model_traced(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Traced model evaluation step."""
    return evaluate_xgb_model(model, X_test, y_test)

@mlflow.trace(attributes={"step": "model_logging"})
def log_model_traced(model: Any, model_version: str) -> None:
    """Traced model logging step."""
    # Ensure the artifacts/models directory exists
    models_dir = os.path.join("artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model to the artifacts/models directory
    model_path = os.path.join(models_dir, f"model_v{model_version}.joblib")
    mlflow.xgboost.log_model(
        model,
        "model",
        registered_model_name=f"churn_model_v{model_version}"
    )
    save_xgb_model(model, model_path)

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

    # Save detailed info as JSON artifact in artifacts/system/
    system_dir = os.path.join("artifacts", "system")
    os.makedirs(system_dir, exist_ok=True)
    system_path = os.path.join(system_dir, "system_info.json")
    with open(system_path, "w") as f:
        json.dump(system_info, f, indent=4)
    mlflow.log_artifact(system_path)
    print("✓ System information logged")

def handle_error(run, e):
    """Handle errors and log error information as an artifact."""
    error_info = {
        "error_type": str(type(e).__name__),
        "error_message": str(e),
        "timestamp": datetime.now().isoformat()
    }

    # Save error info as JSON artifact in artifacts/errors/
    errors_dir = os.path.join("artifacts", "errors")
    os.makedirs(errors_dir, exist_ok=True)
    error_path = os.path.join(errors_dir, "error_info.json")
    with open(error_path, "w") as f:
        json.dump(error_info, f, indent=4)
    mlflow.log_artifact(error_path)

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the enhanced ML pipeline with comprehensive MLflow tracking and tracing."""
    print("🚀 Launching enhanced ML pipeline...")

    experiment_id = setup_enhanced_mlflow()
    model_version = datetime.now().strftime("%Y%m%d_%H%M")

    with mlflow.start_run(run_name=f"Pipeline_v{model_version}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        try:
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
            print("📊 Preparing data...")
            X_train, X_test, y_train, y_test = prepare_data_traced(train_file, test_file)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            print("✅ Data preparation complete")

            # Model training phase
            print("🔧 Training model...")
            model = train_model_traced(X_train, y_train, model_version=model_version)
            print("✅ Model training complete")

            # Model evaluation phase
            print("📈 Evaluating model...")
            metrics = evaluate_model_traced(model, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(f"metric.{k}", float(v))
            print("✅ Model evaluation complete")

            # Log model
            log_model_traced(model, model_version)

            # Log completion
            mlflow.log_param("completion_time", datetime.now().isoformat())
            mlflow.set_tag("pipeline_status", "completed")

            artifact_uri = run.info.artifact_uri
            print(f"✨ Pipeline completed successfully - Model Version: {model_version}")
            print(f"📁 Artifacts saved to: {artifact_uri}")
            print(f"🔍 MLflow UI: http://localhost:5001")

        except Exception as e:
            handle_error(run, e)
            print(f"❌ Error in pipeline: {str(e)}")
            raise

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
                X_train, _, y_train, _ = prepare_data_traced(args.train_file, args.test_file)
                train_model_traced(X_train, y_train, model_version=datetime.now().strftime("%Y%m%d_%H%M"))
        elif args.action == "evaluate":
            with mlflow.start_run() as run:
                model = load_model()  # Use the original load_model since tracing isn't needed here
                _, X_test, _, y_test = prepare_data_traced(args.train_file, args.test_file)
                evaluate_model_traced(model, X_test, y_test)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
