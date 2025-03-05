"""Module for retraining machine learning models with proper tracking and promotion."""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from mlflow.tracking import MlflowClient

# Import from project modules
from data_processing import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model
from model_persistence import save_model
from db_connector import get_db_connector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow_tracking():
    """Set up MLflow tracking environment."""
    try:
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = "churn_prediction"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.abspath("./artifacts/mlruns"),
            )
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"Failed to set up MLflow tracking: {e}")
        return None

def retrain_model(
    train_file: str,
    test_file: str,
    auto_promote: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Retrain the model with new data.
    
    Args:
        train_file: Path to training data file
        test_file: Path to testing data file
        auto_promote: Whether to automatically promote the model to production
        
    Returns:
        Tuple of (success, result_data)
    """
    try:
        # Generate model version based on timestamp
        model_version = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = "churn_prediction_model"
        result_data = {
            "model_version": model_version,
            "training_started": datetime.now().isoformat(),
            "status": "failed",
            "metrics": {},
            "data_stats": {},
            "errors": []
        }
        
        # Set up MLflow
        experiment_id = setup_mlflow_tracking()
        if not experiment_id:
            result_data["errors"].append("Failed to set up MLflow tracking")
            return False, result_data
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            result_data["run_id"] = run_id
            
            logger.info(f"Started retraining with run_id: {run_id}")
            
            # Prepare data
            logger.info("Preparing data...")
            try:
                X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
                
                # Log data statistics
                result_data["data_stats"] = {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": list(X_train.columns),
                    "feature_stats": {col: {
                        "mean": float(X_train[col].mean()) if pd.api.types.is_numeric_dtype(X_train[col]) else None,
                        "min": float(X_train[col].min()) if pd.api.types.is_numeric_dtype(X_train[col]) else None,
                        "max": float(X_train[col].max()) if pd.api.types.is_numeric_dtype(X_train[col]) else None,
                    } for col in X_train.columns}
                }
            except Exception as e:
                error_msg = f"Data preparation failed: {str(e)}"
                logger.error(error_msg)
                result_data["errors"].append(error_msg)
                return False, result_data
            
            # Train model
            logger.info("Training model...")
            try:
                model = train_model(X_train, y_train, model_version=model_version)
                
                # Add model info for later retrieval
                model.model_info = {
                    "version": model_version,
                    "train_date": datetime.now().isoformat(),
                    "features": list(X_train.columns),
                    "training_samples": len(X_train),
                    "stage": "Development"
                }
                
                # Save trained model
                model_filename = os.path.join(
                    "artifacts", "models", f"model_{model_version}.joblib"
                )
                save_model(model, model_filename)
                logger.info(f"Model saved to {model_filename}")
            except Exception as e:
                error_msg = f"Model training failed: {str(e)}"
                logger.error(error_msg)
                result_data["errors"].append(error_msg)
                return False, result_data
            
            # Evaluate model
            logger.info("Evaluating model...")
            try:
                metrics = evaluate_model(model, X_test, y_test)
                result_data["metrics"] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in metrics.items()}
                logger.info(f"Evaluation metrics: {metrics}")
                
                # Log metrics to MongoDB if available
                db = get_db_connector()
                if db:
                    db.save_model_metrics(model_version, result_data["metrics"])
                    logger.info("Saved metrics to MongoDB")
            except Exception as e:
                error_msg = f"Model evaluation failed: {str(e)}"
                logger.error(error_msg)
                result_data["errors"].append(error_msg)
                return False, result_data
            
            # Register model with MLflow
            logger.info("Registering model with MLflow...")
            try:
                # Infer model signature
                signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
                
                # Log the model with details
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=X_train.iloc[:5],
                    registered_model_name=f"{model_name}_v{model_version}",
                )
                
                logger.info(f"Successfully registered model '{model_name}_v{model_version}'")
                
                # Get the MLflow client
                client = MlflowClient()
                
                # Get the latest version of the model
                model_versions = client.search_model_versions(f"name='{model_name}_v{model_version}'")
                if model_versions:
                    latest_version = model_versions[0]
                    version_num = latest_version.version
                    
                    # Transition the model to staging
                    client.transition_model_version_stage(
                        name=f"{model_name}_v{model_version}",
                        version=version_num,
                        stage="Staging"
                    )
                    
                    logger.info(f"Transitioned model '{model_name}_v{model_version}' version {version_num} to 'Staging'")
                    result_data["registry_version"] = version_num
                    model.model_info["stage"] = "Staging"
                    save_model(model, model_filename) # Update the model info
                
            except Exception as e:
                error_msg = f"Model registration failed: {str(e)}"
                logger.error(error_msg)
                result_data["errors"].append(error_msg)
                # Continue without failing, as the model is still saved locally
            
            # Promote model if requested and metrics are good
            if auto_promote and "accuracy" in metrics and "roc_auc" in metrics:
                logger.info("Checking if model should be promoted to production...")
                ACCURACY_THRESHOLD = 0.85
                ROC_AUC_THRESHOLD = 0.80
                
                should_promote = (metrics.get('accuracy', 0) >= ACCURACY_THRESHOLD and 
                                metrics.get('roc_auc', 0) >= ROC_AUC_THRESHOLD)
                
                result_data["promotion_eligible"] = should_promote
                result_data["promotion_thresholds"] = {
                    "accuracy": ACCURACY_THRESHOLD,
                    "roc_auc": ROC_AUC_THRESHOLD
                }
                
                if should_promote:
                    try:
                        # Get the MLflow client
                        client = MlflowClient()
                        
                        # Archive previous production models
                        production_versions = [
                            mv for mv in client.search_model_versions(f"name='{model_name}_v{model_version}'") 
                            if mv.current_stage == "Production"
                        ]
                        
                        for mv in production_versions:
                            client.transition_model_version_stage(
                                name=f"{model_name}_v{model_version}",
                                version=mv.version,
                                stage="Archived"
                            )
                            logger.info(f"Archived previous production model {model_name}_v{model_version} version {mv.version}")
                        
                        # Promote current model
                        client.transition_model_version_stage(
                            name=f"{model_name}_v{model_version}",
                            version=version_num,
                            stage="Production"
                        )
                        
                        # Update description with metrics
                        client.update_model_version(
                            name=f"{model_name}_v{model_version}",
                            version=version_num,
                            description=f"Production model with accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}"
                        )
                        
                        logger.info(f"Promoted model '{model_name}_v{model_version}' version {version_num} to Production")
                        result_data["promoted_to_production"] = True
                        
                        # Update model info and save again
                        model.model_info["stage"] = "Production"
                        model.model_info["promotion_date"] = datetime.now().isoformat()
                        save_model(model, model_filename)
                        
                    except Exception as e:
                        error_msg = f"Model promotion failed: {str(e)}"
                        logger.error(error_msg)
                        result_data["errors"].append(error_msg)
                        result_data["promoted_to_production"] = False
                else:
                    logger.info(f"Model did not meet promotion criteria: accuracy={metrics.get('accuracy', 0):.4f}, roc_auc={metrics.get('roc_auc', 0):.4f}")
                    result_data["promotion_eligible"] = False
            
            # Set result status to success if no errors
            if not result_data["errors"]:
                result_data["status"] = "success"
                result_data["training_completed"] = datetime.now().isoformat()
            
            return True, result_data
    
    except Exception as e:
        logger.error(f"Retraining failed with unexpected error: {str(e)}")
        return False, {
            "status": "failed",
            "errors": [f"Unexpected error: {str(e)}"],
            "model_version": datetime.now().strftime("%Y%m%d_%H%M") if 'model_version' not in locals() else model_version
        }

def get_available_datasets():
    """
    List available datasets that can be used for retraining.
    
    Returns:
        Dict of available datasets
    """
    try:
        datasets = []
        
        # Look for CSV files in the current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=5)  # Read just a few rows to get columns
                has_target = "Churn" in df.columns
                datasets.append({
                    "filename": csv_file,
                    "has_target": has_target,
                    "columns": list(df.columns),
                    "rows_preview": len(pd.read_csv(csv_file, nrows=0)),
                    "location": "root",
                    "recommended_for": "train" if "bigml-80" in csv_file else "test" if "bigml-20" in csv_file else "unknown"
                })
            except Exception as e:
                logger.warning(f"Could not analyze dataset {csv_file}: {e}")
        
        # Look for CSV files in the artifacts/data directory
        artifacts_data_dir = os.path.join("artifacts", "data")
        if os.path.exists(artifacts_data_dir):
            csv_files = [f for f in os.listdir(artifacts_data_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(artifacts_data_dir, csv_file)
                    df = pd.read_csv(file_path, nrows=5)  # Read just a few rows to get columns
                    has_target = "Churn" in df.columns
                    datasets.append({
                        "filename": file_path,
                        "has_target": has_target,
                        "columns": list(df.columns),
                        "rows_preview": len(pd.read_csv(file_path, nrows=0)),
                        "location": "artifacts/data",
                        "recommended_for": "train" if "train" in csv_file else "test" if "test" in csv_file else "unknown"
                    })
                except Exception as e:
                    logger.warning(f"Could not analyze dataset {file_path}: {e}")
        
        return {
            "status": "success", 
            "datasets": datasets
        }
    except Exception as e:
        logger.error(f"Error getting available datasets: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_registered_models():
    """
    Get information about registered models.
    
    Returns:
        Dict of registered models
    """
    try:
        # Set up MLflow
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get the MLflow client
        client = MlflowClient()
        
        # Get all registered models
        registered_models = client.search_registered_models()
        
        models_info = []
        for rm in registered_models:
            model_name = rm.name
            
            # Get latest versions of the model
            versions = client.search_model_versions(f"name='{model_name}'")
            versions_info = []
            
            for v in versions:
                # Get run info
                try:
                    run = client.get_run(v.run_id)
                    metrics = run.data.metrics
                    params = run.data.params
                except:
                    metrics = {}
                    params = {}
                
                versions_info.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                    "last_updated_timestamp": v.last_updated_timestamp,
                    "description": v.description,
                    "metrics": metrics,
                    "params": params
                })
            
            models_info.append({
                "name": model_name,
                "versions": versions_info
            })
        
        return {
            "status": "success",
            "models": models_info
        }
    except Exception as e:
        logger.error(f"Error getting registered models: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def manual_promote_model(model_name, version):
    """
    Manually promote a model version to production.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote
        
    Returns:
        Dict with promotion result
    """
    try:
        # Set up MLflow
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get the MLflow client
        client = MlflowClient()
        
        # Archive previous production models
        production_versions = [
            mv for mv in client.search_model_versions(f"name='{model_name}'") 
            if mv.current_stage == "Production"
        ]
        
        for mv in production_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )
            logger.info(f"Archived previous production model {model_name} version {mv.version}")
        
        # Promote specified version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=version,
            description=f"Manually promoted to Production on {datetime.now().isoformat()}"
        )
        
        logger.info(f"Promoted model '{model_name}' version {version} to Production")
        
        # Try to update the model file
        try:
            model_version = model_name.replace("churn_prediction_model_v", "")
            model_path = os.path.join("artifacts", "models", f"model_{model_version}.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                if hasattr(model, "model_info"):
                    model.model_info["stage"] = "Production"
                    model.model_info["promotion_date"] = datetime.now().isoformat()
                    save_model(model, model_path)
        except Exception as e:
            logger.warning(f"Could not update model file: {e}")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' version {version} promoted to Production"
        }
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Simple CLI for testing
    if len(sys.argv) < 3:
        print("Usage: python model_retrain.py <train_file> <test_file> [--auto-promote]")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    auto_promote = "--auto-promote" in sys.argv
    
    success, result = retrain_model(train_file, test_file, auto_promote)
    print(json.dumps(result, indent=2))
