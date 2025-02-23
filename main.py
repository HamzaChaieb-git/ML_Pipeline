"""Main module for running the ML pipeline."""

import argparse
import os
import mlflow
import sys
from datetime import datetime
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model

def setup_mlflow():
    """Setup MLflow tracking."""
    # Set the tracking URI to use SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Create or get the experiment
    experiment_name = "churn_prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def get_model_version():
    """Generate model version based on date and time."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M")

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
    # Setup MLflow and get model version
    experiment_id = setup_mlflow()
    model_version = get_model_version()
    
    # Start a new MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Full_Pipeline_v{model_version}") as run:
        try:
            # Log input files
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)
            mlflow.log_param("model_version", model_version)
            
            # Data preparation
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            print("🔹 Data preparation complete")
            
            # Model training with version
            print("🔹 Training model...")
            model = train_xgb_model(X_train, y_train, model_version=model_version)
            print("🔹 Model training complete")
            
            # Model evaluation
            print("🔹 Evaluating model...")
            metrics = evaluate_xgb_model(model, X_test, y_test)
            print("🔹 Evaluation complete")
            
            # Register model in MLflow Model Registry
            client = mlflow.tracking.MlflowClient()
            
            # Check if metrics meet production criteria
            is_production_ready = (
                metrics.get("accuracy", 0) > 0.95 and 
                metrics.get("roc_auc", 0) > 0.90
            )
            
            # Set the appropriate stage
            stage = "Production" if is_production_ready else "Staging"
            
            try:
                # Try to update model stage
                model_details = client.get_latest_versions(f"churn_prediction_model_v{model_version}", stages=["None"])
                if model_details:
                    client.transition_model_version_stage(
                        name=f"churn_prediction_model_v{model_version}",
                        version=model_details[0].version,
                        stage=stage
                    )
                    print(f"🔹 Model transitioned to {stage} stage")
                
                # Log model artifacts and metadata
                if os.path.exists("training_curve_final.png"):
                    mlflow.log_artifact("training_curve_final.png", "training_curves")
                if os.path.exists("feature_importance.png"):
                    mlflow.log_artifact("feature_importance.png", "feature_importance")
                
                print(f"🔹 Model version {model_version} registered successfully")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not update model stage: {str(e)}")
            
            return model, metrics
                
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e

def main() -> None:
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument(
        "action",
        type=str,
        nargs="?",
        default="all",
        help="Action to perform: prepare_data, train_model, evaluate_model, save_model, or run all steps."
    )
    args = parser.parse_args()
    
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    try:
        if args.action == "all":
            run_full_pipeline(train_file, test_file)
        else:
            print("\n❌ Invalid action! Choose 'all' to run the complete pipeline.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
