"""Enhanced main module for running the ML pipeline with comprehensive MLflow tracking."""

import argparse
import os
import mlflow
import sys
from datetime import datetime
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model
import json

def setup_enhanced_mlflow():
    """Setup enhanced MLflow tracking with custom configuration."""
    # Set up SQLite tracking
    mlflow.set_tracking_uri("sqlite:///enhanced_mlflow.db")
    
    # Create or get the experiment with custom metadata
    experiment_name = "enhanced_churn_prediction"
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
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location="./mlruns/enhanced_experiments",
                tags=experiment_tags
            )
    except Exception as e:
        print(f"⚠️ Warning: Error setting up experiment: {str(e)}")
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def get_enhanced_model_version():
    """Generate detailed model version with timestamp and hash."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"{timestamp}"

def log_system_info():
    """Log system and environment information."""
    import platform
    import psutil
    
    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "processor": platform.processor(),
        "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
        "memory_available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq().max if hasattr(psutil.cpu_freq(), 'max') else None
    }
    
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=4)
    mlflow.log_artifact("system_info.json")

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the enhanced ML pipeline with comprehensive MLflow tracking."""
    print("🚀 Launching enhanced ML pipeline...")
    
    # Setup MLflow with enhanced configuration
    experiment_id = setup_enhanced_mlflow()
    model_version = get_enhanced_model_version()
    
    # Start MLflow run with nested runs for each phase
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Enhanced_Pipeline_v{model_version}") as main_run:
        try:
            # Log system information
            log_system_info()
            
            # Log input parameters
            mlflow.log_params({
                "train_file": train_file,
                "test_file": test_file,
                "model_version": model_version,
                "pipeline_type": "enhanced"
            })
            
            # Data preparation phase
            with mlflow.start_run(run_name="data_preparation", nested=True):
                print("📊 Preparing data...")
                mlflow.log_param("data_preparation_start", datetime.now().isoformat())
                X_train, X_test, y_train, y_test = process_data(train_file, test_file)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("test_samples", len(X_test))
                print("✅ Data preparation complete")
            
            # Model training phase
            with mlflow.start_run(run_name="model_training", nested=True):
                print("🔧 Training model...")
                mlflow.log_param("training_start", datetime.now().isoformat())
                model = train_xgb_model(X_train, y_train, model_version=model_version)
                print("✅ Model training complete")
            
            # Model evaluation phase
            with mlflow.start_run(run_name="model_evaluation", nested=True):
                print("📈 Evaluating model...")
                mlflow.log_param("evaluation_start", datetime.now().isoformat())
                metrics = evaluate_xgb_model(model, X_test, y_test)
                print("✅ Model evaluation complete")
            
            # Model registration and staging
            client = mlflow.tracking.MlflowClient()
            
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
                # Update model stage
                model_details = client.get_latest_versions(
                    f"churn_prediction_model_v{model_version}",
                    stages=["None"]
                )
                if model_details:
                    client.transition_model_version_stage(
                        name=f"churn_prediction_model_v{model_version}",
                        version=model_details[0].version,
                        stage=stage
                    )
                    print(f"✅ Model transitioned to {stage} stage")
                
                # Log final run status
                run_info = {
                    "status": "completed",
                    "completion_time": datetime.now().isoformat(),
                    "model_stage": stage,
                    "is_production_ready": is_production_ready,
                    "model_version": model_version
                }
                
                with open("run_info.json", "w") as f:
                    json.dump(run_info, f, indent=4)
                mlflow.log_artifact("run_info.json")
                
                print(f"✨ Pipeline completed successfully - Model Version: {model_version}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not update model stage: {str(e)}")
            
            return model, metrics
                
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
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
