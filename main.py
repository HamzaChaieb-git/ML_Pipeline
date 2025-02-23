"""Main module for running the ML pipeline."""

import argparse
import os
import mlflow
import sys
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
    # Start a new MLflow run
    with mlflow.start_run(run_name="Full Pipeline") as run:
        try:
            # Log input parameters
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)
            
            # Data preparation
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            print("🔹 Data preparation complete")
            
            # Model training
            print("🔹 Training model...")
            model = train_xgb_model(X_train, y_train)
            print("🔹 Model training complete")
            
            # Model evaluation
            print("🔹 Evaluating model...")
            metrics = evaluate_xgb_model(model, X_test, y_test)
            print("🔹 Evaluation complete")
            
            # Save model
            print("🔹 Saving model...")
            save_xgb_model(model)
            mlflow.log_artifact("model.joblib")
            print("🔹 Model saved")
            
            # Load and evaluate again
            print("🔹 Loading model...")
            loaded_model = load_xgb_model()
            print("🔹 Model loaded")
            
            # Final evaluation
            print("🔹 Final evaluation...")
            final_metrics = evaluate_xgb_model(loaded_model, X_test, y_test)
            
            # Log any additional artifacts
            if os.path.exists("feature_importance.json"):
                mlflow.log_artifact("feature_importance.json")
            if os.path.exists("confusion_matrix.csv"):
                mlflow.log_artifact("confusion_matrix.csv")
            
            return model, metrics
                
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e

def main() -> None:
    """Main function to run the pipeline."""
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"
    
    try:
        run_full_pipeline(train_file, test_file)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
