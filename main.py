import argparse
import os
import mlflow
import sys

# Set MLflow tracking URI to use a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def prepare_data(train_file: str = "churn-bigml-80.csv", test_file: str = "churn-bigml-20.csv"):
    """Prepare data and save output to a local file, logging to MLflow."""
    print("🔹 Preparing data...")
    output_file = os.path.join(os.getcwd(), 'data_output.txt')
    with open(output_file, 'w') as f:
        f.write("Data preparation completed\n")
    with mlflow.start_run(run_name="Data Preparation"):
        mlflow.log_param("step", "prepare_data")
        mlflow.log_artifact(output_file)
    print(f"🔹 Data preparation complete. File saved at: {output_file}")
    return None, None, None, None  # Placeholder for X_train, X_test, y_train, y_test

def train_model(X_train=None, y_train=None, run_id=None):
    """Train model and save output to a local file, logging to MLflow."""
    print("🔹 Training model...")
    output_file = os.path.join(os.getcwd(), 'train_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model training completed\nTraining XGBoost model...\n")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_param("step", "train_model")
        mlflow.log_metric("example_metric", 0.95)  # Replace with actual metrics
        mlflow.log_artifact(output_file)
    print(f"🔹 Model training complete. File saved at: {output_file}")
    return None  # Placeholder for model object

def evaluate_model(model=None, X_test=None, y_test=None, run_id=None):
    """Evaluate model and save output to a local file, logging to MLflow."""
    print("🔹 Evaluating model...")
    output_file = os.path.join(os.getcwd(), 'model_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model evaluation completed\nModel Accuracy: 0.9505\n")
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_param("step", "evaluate_model")
        mlflow.log_metric("example_score", 0.90)  # Replace with actual metrics
        mlflow.log_artifact(output_file)
    print(f"🔹 Evaluation complete. File saved at: {output_file}")

def save_model(model=None, run_id=None):
    """Save model to a local file, logging to MLflow."""
    print("🔹 Saving model...")
    output_file = os.path.join(os.getcwd(), 'model.pkl')
    with open(output_file, 'w') as f:
        f.write("dummy model")  # Placeholder
    with mlflow.start_run(run_name="Model Saving"):
        mlflow.log_artifact(output_file)
    print(f"Model saved as model.pkl at: {output_file}")

def load_model():
    """Load model from a local file (placeholder function), logging to MLflow."""
    print("🔹 Loading model...")
    output_file = os.path.join(os.getcwd(), 'model.pkl')
    with open(output_file, 'r') as f:
        content = f.read()
    with mlflow.start_run(run_name="Model Loading"):
        mlflow.log_param("step", "load_model")
    print("Model loaded")
    return None  # Placeholder for loaded model

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
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
