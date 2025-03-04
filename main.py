"""Enhanced main module for running the ML pipeline with MLflow native tracing, model registry, and monitoring."""

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
from mlflow.tracking import MlflowClient

# Import monitoring if available
try:
    from monitoring import log_model_metrics, setup_monitoring_db
    from db_connector import get_db_connector
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False

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
        "pipeline_type": "binary_classification",
    }

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.abspath("./artifacts/mlruns"),
            tags=experiment_tags,
        )
        print(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


@mlflow.trace
def prepare_data_traced(train_file: str, test_file: str) -> Tuple:
    """Traced data preparation step."""
    X_train, X_test, y_train, y_test = process_data(train_file, test_file)

    integer_columns = [
        "Customer service calls",
        "Total intl calls",
        "Number vmail messages",
    ]
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


def register_model_with_stages(model, model_name, X_train, model_version):
    """
    Register model with MLflow Model Registry and set up staging.
    
    Args:
        model: Trained model object
        model_name: Base name for the registered model
        X_train: Training data to infer model signature
        model_version: Version string for this model
    
    Returns:
        Registered model version object
    """
    # Get the current MLflow run
    current_run = mlflow.active_run()
    if not current_run:
        print("No active MLflow run. Starting a new run...")
        current_run = mlflow.start_run()
    
    # Infer model signature
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    
    # Log the model with details
    mlflow.xgboost.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:5] if hasattr(X_train, "iloc") else X_train[:5],
        registered_model_name=f"{model_name}_v{model_version}",
    )
    
    print(f"Successfully registered model '{model_name}_v{model_version}'.")
    
    # Get the MLflow client
    client = MlflowClient()
    
    # Get the latest version of the model
    model_versions = client.search_model_versions(f"name='{model_name}_v{model_version}'")
    if model_versions:
        latest_version = model_versions[0]
        version_num = latest_version.version
        print(f"Created version '{version_num}' of model '{model_name}_v{model_version}'.")
        
        # Transition the model to staging
        client.transition_model_version_stage(
            name=f"{model_name}_v{model_version}",
            version=version_num,
            stage="Staging"
        )
        print(f"Transitioned model '{model_name}_v{model_version}' version {version_num} to 'Staging'.")
        
        return latest_version
    
    return None


def promote_model_to_production(model_name, version, model_metrics):
    """
    Promote a model to production if it meets performance criteria.
    
    Args:
        model_name: Name of the registered model
        version: Version number to promote
        model_metrics: Dict containing model performance metrics
    
    Returns:
        True if promotion succeeded, False otherwise
    """
    # Define performance thresholds for production
    ACCURACY_THRESHOLD = 0.90
    ROC_AUC_THRESHOLD = 0.85
    
    # Check if metrics meet thresholds
    if (model_metrics.get('accuracy', 0) >= ACCURACY_THRESHOLD and 
        model_metrics.get('roc_auc', 0) >= ROC_AUC_THRESHOLD):
        try:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            # Add additional metadata to model
            client.update_model_version(
                name=model_name,
                version=version,
                description=f"Production model with accuracy: {model_metrics['accuracy']:.4f}, ROC AUC: {model_metrics['roc_auc']:.4f}"
            )
            
            # Save promotion record for auditing
            promotion_record = {
                "model_name": model_name,
                "version": version,
                "promotion_time": datetime.now().isoformat(),
                "metrics": model_metrics,
                "promoted_by": "automated_pipeline"
            }
            
            os.makedirs("artifacts/model_registry", exist_ok=True)
            with open(f"artifacts/model_registry/promotion_{model_name}_v{version}.json", "w") as f:
                json.dump(promotion_record, f, indent=2)
            
            print(f"✅ Model {model_name} version {version} promoted to Production!")
            return True
        except Exception as e:
            print(f"❌ Failed to promote model to Production: {e}")
            return False
    else:
        print(f"⚠️ Model {model_name} version {version} did not meet production criteria.")
        print(f"   Required: Accuracy >= {ACCURACY_THRESHOLD}, ROC AUC >= {ROC_AUC_THRESHOLD}")
        print(f"   Actual: Accuracy = {model_metrics.get('accuracy', 0):.4f}, ROC AUC = {model_metrics.get('roc_auc', 0):.4f}")
        return False


def archive_previous_production_models(model_name):
    """Archive any previous production models for the given model name."""
    client = MlflowClient()
    
    # Get production versions
    production_versions = [
        mv for mv in client.search_model_versions(f"name='{model_name}'") 
        if mv.current_stage == "Production"
    ]
    
    # Archive old production versions
    for mv in production_versions:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )
            print(f"Archived previous production model {model_name} version {mv.version}")
        except Exception as e:
            print(f"Failed to archive model {model_name} version {mv.version}: {e}")


def store_model_metrics_in_db(model_version, metrics):
    """Store model metrics in monitoring database if available."""
    if not MONITORING_ENABLED:
        return
    
    try:
        # Initialize monitoring database
        engine = setup_monitoring_db()
        # Log metrics
        log_model_metrics(engine, model_version, metrics)
        print(f"Logged model metrics to monitoring database for version {model_version}")
        
        # Also log to MongoDB if available
        db = get_db_connector()
        if db:
            db.save_model_metrics(model_version, metrics)
            print(f"Logged model metrics to MongoDB for version {model_version}")
    except Exception as e:
        print(f"Failed to log metrics to database: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the ML pipeline with specified action"
    )
    parser.add_argument("--train-file", required=True, help="Path to training data CSV")
    parser.add_argument("--test-file", required=True, help="Path to testing data CSV")
    parser.add_argument(
        "--action",
        choices=["train", "evaluate", "all"],
        default="all",
        help="Action to perform: 'train', 'evaluate', or 'all'",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Automatically promote model to production if metrics meet thresholds",
    )
    args = parser.parse_args()

    # Set up MLflow
    experiment_id = setup_enhanced_mlflow()

    # Generate model version based on timestamp
    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = "churn_prediction_model"

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data_traced(
            args.train_file, args.test_file
        )

        model = None
        if args.action in ["train", "all"]:
            # Train model
            model = train_model_traced(X_train, y_train, model_version)
            model_filename = os.path.join(
                "artifacts", "models", f"model_{model_version}.joblib"
            )
            
            # Add model info for later retrieval
            model.model_info = {
                "version": model_version,
                "train_date": datetime.now().isoformat(),
                "features": list(X_train.columns),
                "training_samples": len(X_train)
            }
            
            save_xgb_model(model, model_filename)
            
            # Register model with MLflow registry
            register_model_with_stages(model, model_name, X_train, model_version)

        if args.action in ["evaluate", "all"]:
            # Load model if not trained in this run
            if model is None:
                try:
                    models_dir = os.path.join("artifacts", "models")
                    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_")]
                    
                    if not model_files:
                        raise FileNotFoundError("No model files found for evaluation")
                    
                    latest_model = max(
                        model_files,
                        key=lambda x: os.path.getctime(os.path.join(models_dir, x))
                    )
                    
                    model_path = os.path.join(models_dir, latest_model)
                    model = load_xgb_model(model_path)
                    
                    # Extract version from filename
                    model_version = latest_model.replace("model_", "").replace(".joblib", "")
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise FileNotFoundError(f"No model found for evaluation: {e}")

            # Evaluate model
            metrics = evaluate_model_traced(model, X_test, y_test)
            print(f"Evaluation completed with metrics: {metrics}")
            
            # Store metrics in monitoring database
            store_model_metrics_in_db(model_version, metrics)
            
            # Promote model to production if requested and metrics meet thresholds
            if args.promote:
                # Archive any existing production models first
                archive_previous_production_models(f"{model_name}_v{model_version}")
                
                # Promote current model if it meets criteria
                promoted = promote_model_to_production(
                    f"{model_name}_v{model_version}",
                    "1",  # First version after registration
                    metrics
                )
                
                if promoted:
                    # Update model file with production status
                    model.model_info["stage"] = "Production"
                    model.model_info["promotion_date"] = datetime.now().isoformat()
                    model.model_info["performance_metrics"] = metrics
                    
                    # Save updated model
                    model_filename = os.path.join(
                        "artifacts", "models", f"model_{model_version}.joblib"
                    )
                    save_xgb_model(model, model_filename)


if __name__ == "__main__":
    main()