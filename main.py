"""Enhanced main module for running the ML pipeline with MLflow and OpenTelemetry tracking."""

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

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.status import Status, StatusCode

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add ConsoleSpanExporter
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

def setup_enhanced_mlflow():
    """Setup enhanced MLflow tracking with custom configuration."""
    with tracer.start_as_current_span("setup_mlflow") as span:
        # Set up MLflow tracking URI to match existing configuration
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")
        span.set_attribute("mlflow.tracking_uri", tracking_uri)
        
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
            span.set_attribute("mlflow.experiment_id", experiment_id)
            return experiment_id
        except Exception as e:
            print(f"⚠️ Warning: Error setting up experiment: {str(e)}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

def log_system_info():
    """Log system and environment information."""
    with tracer.start_as_current_span("log_system_info") as span:
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
                span.set_attribute(f"system.{key}", str(value))
        
        # Save detailed info as JSON artifact
        with open("system_info.json", "w") as f:
            json.dump(system_info, f, indent=4)
        mlflow.log_artifact("system_info.json")
        print("✓ System information logged")

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the enhanced ML pipeline with comprehensive MLflow tracking."""
    with tracer.start_as_current_span("pipeline_execution") as span:
        print("🚀 Launching enhanced ML pipeline...")
        
        try:
            # Setup MLflow with enhanced configuration
            experiment_id = setup_enhanced_mlflow()
            model_version = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"Pipeline_v{model_version}") as run:
                # Log run information
                run_id = run.info.run_id
                artifact_uri = run.info.artifact_uri
                print(f"MLflow Run ID: {run_id}")
                print(f"Artifact URI: {artifact_uri}")
                
                span.set_attribute("mlflow.run_id", run_id)
                span.set_attribute("mlflow.artifact_uri", artifact_uri)
                
                # Log system information
                with tracer.start_as_current_span("system_info_logging"):
                    log_system_info()
                
                # Log input parameters
                mlflow.log_params({
                    "train_file": train_file,
                    "test_file": test_file,
                    "model_version": model_version,
                    "pipeline_type": "enhanced"
                })
                
                # Data preparation phase
                with tracer.start_as_current_span("data_preparation") as data_span:
                    print("📊 Preparing data...")
                    X_train, X_test, y_train, y_test = process_data(train_file, test_file)
                    mlflow.log_metric("train_samples", len(X_train))
                    mlflow.log_metric("test_samples", len(X_test))
                    data_span.set_attribute("train_samples", len(X_train))
                    data_span.set_attribute("test_samples", len(X_test))
                    print("✅ Data preparation complete")
                
                # Model training phase
                with tracer.start_as_current_span("model_training") as training_span:
                    print("🔧 Training model...")
                    model = train_xgb_model(X_train, y_train, model_version=model_version)
                    print("✅ Model training complete")
                
                # Model evaluation phase
                with tracer.start_as_current_span("model_evaluation") as eval_span:
                    print("📈 Evaluating model...")
                    metrics = evaluate_xgb_model(model, X_test, y_test)
                    eval_span.set_attributes({
                        f"metric.{k}": float(v) for k, v in metrics.items()
                    })
                    print("✅ Model evaluation complete")
                
                # Log model
                with tracer.start_as_current_span("model_logging"):
                    mlflow.xgboost.log_model(
                        model,
                        "model",
                        registered_model_name=f"churn_model_v{model_version}"
                    )
                    
                    # Save model locally
                    save_xgb_model(model, f"model_v{model_version}.joblib")
                
                # Log completion
                mlflow.log_param("completion_time", datetime.now().isoformat())
                mlflow.set_tag("pipeline_status", "completed")
                
                print(f"✨ Pipeline completed successfully - Model Version: {model_version}")
                print(f"📁 Artifacts saved to: {artifact_uri}")
                print(f"🔍 MLflow UI: http://localhost:5001")
                
                span.set_status(Status(StatusCode.OK))
                return model, metrics
                
        except Exception as e:
            error_info = {
                "error_type": str(type(e).__name__),
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            with open("error_info.json", "w") as f:
                json.dump(error_info, f, indent=4)
            mlflow.log_artifact("error_info.json")
            
            span.set_status(Status(StatusCode.ERROR, str(e)))
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
