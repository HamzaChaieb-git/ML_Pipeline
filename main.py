import argparse
import os
import mlflow
import sys
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model

# Set MLflow tracking URI to use a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def prepare_data(train_file: str = "churn-bigml-80.csv", test_file: str = "churn-bigml-20.csv"):
    """Prepare data using the actual implementation."""
    print("🔹 Preparing data...")
    X_train, X_test, y_train, y_test = process_data(train_file, test_file)
    with mlflow.start_run(run_name="Data Preparation"):
        mlflow.log_param("train_file", train_file)
        mlflow.log_param("test_file", test_file)
    print("🔹 Data preparation complete")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, run_id=None):
    """Train model using the actual implementation."""
    print("🔹 Training model...")
    with mlflow.start_run(run_name="Model Training"):
        model = train_xgb_model(X_train, y_train)
    print("🔹 Model training complete")
    return model

def evaluate_model(model, X_test, y_test, run_id=None):
    """Evaluate model using the actual implementation."""
    print("🔹 Evaluating model...")
    with mlflow.start_run(run_name="Model Evaluation"):
        evaluate_xgb_model(model, X_test, y_test)
    print("🔹 Evaluation complete")

def save_model(model, run_id=None):
    """Save model using the actual implementation."""
    print("🔹 Saving model...")
    save_xgb_model(model)
    print("🔹 Model saved")

def load_model():
    """Load model using the actual implementation."""
    print("🔹 Loading model...")
    model = load_xgb_model()
    print("🔹 Model loaded")
    return model

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
    with mlflow.start_run(run_name="Full Pipeline") as run:
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model)
        loaded_model = load_model()
        evaluate_model(loaded_model, X_test, y_test)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for pipeline actions."""
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline Controller")
    parser.add_argument(
        "action",
        type=str,
        nargs="?",
        default="all",
        help="Action to perform: prepare_data, train_model, evaluate_model, save_model, load_model, or run all steps by default."
    )
    return parser.parse_args()

def main() -> None:
    """Main function to run the pipeline based on command-line arguments."""
    args = parse_arguments()
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    if args.action == "prepare_data":
        prepare_data(train_file, test_file)
    elif args.action == "train_model":
        with mlflow.start_run() as run:
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            train_model(X_train, y_train, run_id=run.info.run_id)
    elif args.action == "evaluate_model":
        with mlflow.start_run() as run:
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train, run_id=run.info.run_id)
            evaluate_model(model, X_test, y_test, run_id=run.info.run_id)
    elif args.action == "save_model":
        with mlflow.start_run() as run:
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train, run_id=run.info.run_id)
            save_model(model, run_id=run.info.run_id)
    elif args.action == "load_model":
        with mlflow.start_run() as run:
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            loaded_model = load_model()
            evaluate_model(loaded_model, X_test, y_test, run_id=run.info.run_id)
    elif args.action == "all":
        run_full_pipeline(train_file, test_file)
    else:
        print("\n❌ Invalid action! Choose from: prepare_data, train_model, evaluate_model, save_model, load_model, or leave blank to run all.")

if __name__ == "__main__":
    main()
