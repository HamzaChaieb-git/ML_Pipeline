"""Enhanced main module for running the ML pipeline with MLflow tracking and tracing."""

import argparse
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import sys
from datetime import datetime
from typing import Dict, Any, Tuple
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model
import json
import platform
import psutil

def setup_enhanced_mlflow():
    """Setup enhanced MLflow tracking with custom configuration and tracing."""
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
        "pipeline_type": "binary_classification",
        "enable_tracing": "true"
    }
    
    try:
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
        
        # Enable tracing flags
        os.environ["MLFLOW_ENABLE_TRACING"] = "true"
        os.environ["MLFLOW_TRACING_SAMPLING_RATE"] = "1.0"
        
        return experiment_id
    except Exception as e:
        print(f"⚠️ Warning: Error setting up experiment: {str(e)}")
        raise

def log_system_info():
    """Log system and environment information."""
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

def create_run_report(metrics: Dict, model_version: str, stage: str) -> None:
    """Create and log a comprehensive run report."""
    report = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "stage": stage,
            "metrics_summary": {
                "accuracy": float(metrics.get("accuracy", 0)),
                "precision": float(metrics.get("precision", 0)),
                "recall": float(metrics.get("recall", 0)),
                "f1": float(metrics.get("f1", 0)),
                "roc_auc": float(metrics.get("roc_auc", 0))
            },
            "production_readiness": {
                "accuracy_threshold_met": str(metrics.get("accuracy", 0) > 0.85),
                "roc_auc_threshold_met": str(metrics.get("roc_auc", 0) > 0.85),
                "precision_threshold_met": str(metrics.get("precision", 0) > 0.80),
                "recall_threshold_met": str(metrics.get("recall", 0) > 0.80)
            }
        }
    }
    
    # Save report
    with open("run_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("run_report.json")

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the enhanced ML pipeline with comprehensive MLflow tracking and tracing."""
    print("🚀 Launching enhanced ML pipeline...")
    
    # Setup MLflow with enhanced configuration
    experiment_id = setup_enhanced_mlflow()
    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Start MLflow run with tracing enabled
    with mlflow.start_run(run_name=f"Pipeline_v{model_version}") as run:
        try:
            # Enable detailed tracing for this run
            mlflow.start_trace("pipeline_execution")
            
            # Log run information
            run_id = run.info.run_id
            artifact_uri = run.info.artifact_uri
            print(f"MLflow Run ID: {run_id}")
            print(f"Artifact URI: {artifact_uri}")
            
            # Log system information
            with mlflow.start_trace("system_info"):
                log_system_info()
            
            # Log input parameters
            mlflow.log_params({
                "train_file": train_file,
                "test_file": test_file,
                "model_version": model_version,
                "pipeline_type": "enhanced",
                "tracing_enabled": "true"
            })
            
            # Data preparation phase
            with mlflow.start_trace("data_preparation"):
                print("📊 Preparing data...")
                mlflow.log_param("data_preparation_start", datetime.now().isoformat())
                X_train, X_test, y_train, y_test = process_data(train_file, test_file)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("test_samples", len(X_test))
                print("✅ Data preparation complete")
            
            # Model training phase
            with mlflow.start_trace("model_training"):
                print("🔧 Training model...")
                mlflow.log_param("training_start", datetime.now().isoformat())
                model = train_xgb_model(X_train, y_train, model_version=model_version)
                print("✅ Model training complete")
            
            # Model evaluation phase
            with mlflow.start_trace("model_evaluation"):
                print("📈 Evaluating model...")
                mlflow.log_param("evaluation_start", datetime.now().isoformat())
                metrics = evaluate_xgb_model(model, X_test, y_test)
                print("✅ Model evaluation complete")
            
            # Model registration and staging
            with mlflow.start_trace("model_registration"):
                # Define production readiness criteria
                is_production_ready = (
                    metrics.get("accuracy", 0) > 0.85 and 
                    metrics.get("roc_auc", 0) > 0.85 and
                    metrics.get("precision", 0) > 0.80 and
                    metrics.get("recall", 0) > 0.80
                )
                
                # Set appropriate stage based on metrics
                stage = "Production" if is_production_ready else "Staging"
                
                try:
                    # Create and log run report
                    create_run_report(metrics, model_version, stage)
                    
                    # Log final metrics and artifacts
                    mlflow.log_metrics(metrics)
                    mlflow.xgboost.log_model(
                        model,
                        "model",
                        registered_model_name=f"churn_model_v{model_version}"
                    )
                    
                    # Save model locally
                    save_xgb_model(model, f"model_v{model_version}.joblib")
                    
                    print(f"✨ Pipeline completed successfully - Model Version: {model_version}")
                    print(f"📁 Artifacts saved to: {artifact_uri}")
                    print(f"🔍 MLflow UI: http://localhost:5001")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not register model: {str(e)}")
            
            mlflow.end_trace()  # End the pipeline execution trace
            
            return model, metrics
                
        except Exception as e:
            error_info = {
                "error_type": str(type(e).__name__),
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            with mlflow.start_trace("error_handling"):
                with open("error_info.json", "w") as f:
                    json.dump(error_info, f, indent=4)
                mlflow.log_artifact("error_info.json")
            
            print(f"❌ Error in pipeline: {str(e)}")
            raise e

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
