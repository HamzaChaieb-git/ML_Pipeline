"""Main script to control and orchestrate the machine learning pipeline with MLflow tracking."""

import argparse
import os
import mlflow
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Set MLflow tracking URI to use a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def prepare_data(train_file: str = "churn-bigml-80.csv", test_file: str = "churn-bigml-20.csv"):
    """Prepare data by loading and splitting CSV files, returning train/test splits."""
    print("🔹 Preparing data...")
    # Load data (assuming CSV files exist or use placeholders)
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print("Warning: CSV files not found, using placeholder data.")
        # Placeholder data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Assume 'target' is the target column, features are all others
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    output_file = os.path.join(os.getcwd(), 'data_output.txt')
    with open(output_file, 'w') as f:
        f.write("Data preparation completed\n")
    print(f"🔹 Data preparation complete. File saved at: {output_file}")
    with mlflow.start_run(run_name="Data Preparation"):
        mlflow.log_param("step", "prepare_data")
        mlflow.log_artifact(output_file)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, run_id=None):
    """Train an XGBoost model and save output to a local file, logging to MLflow."""
    print("🔹 Training model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    output_file = os.path.join(os.getcwd(), 'train_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model training completed\nTraining XGBoost model...\n")
    print(f"🔹 Model training complete. File saved at: {output_file}")
    
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_param("step", "train_model")
        mlflow.log_metric("example_metric", 0.95)  # Placeholder, replace with actual metric
        mlflow.log_artifact(output_file)
        mlflow.sklearn.log_model(model, "model")
    return model

def evaluate_model(model, X_test, y_test, run_id=None):
    """Evaluate the model and save output to a local file, logging to MLflow."""
    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    
    output_file = os.path.join(os.getcwd(), 'model_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model evaluation completed\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report(y_test, y_pred)) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n")
    
    print(f"🔹 Evaluation complete. File saved at: {output_file}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_param("step", "evaluate_model")
        mlflow.log_metric("example_score", 0.95)  # Placeholder, replace with actual metric
        mlflow.log_artifact(output_file)

def save_model(model, run_id=None):
    """Save the model to a local file, logging to MLflow."""
    print("🔹 Saving model...")
    output_file = os.path.join(os.getcwd(), 'model.joblib')
    joblib.dump(model, output_file)
    print(f"Model saved as model.joblib at: {output_file}")
    
    with mlflow.start_run(run_name="Model Saving"):
        mlflow.log_artifact(output_file)

def load_model():
    """Load the model from a local file, logging to MLflow."""
    print("🔹 Loading model...")
    output_file = os.path.join(os.getcwd(), 'model.joblib')
    model = joblib.load(output_file)
    print("Model loaded from model.joblib")
    
    with mlflow.start_run(run_name="Model Loading"):
        mlflow.log_param("step", "load_model")
    return model

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
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        train_model(X_train, y_train)
    elif args.action == "evaluate_model":
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    elif args.action == "save_model":
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        model = train_model(X_train, y_train)
        save_model(model)
    elif args.action == "load_model":
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        loaded_model = load_model()
        evaluate_model(loaded_model, X_test, y_test)
    elif args.action == "all":
        run_full_pipeline(train_file, test_file)
    else:
        print("\n❌ Invalid action! Choose from: prepare_data, train_model, evaluate_model, save_model, load_model, or leave blank to run all.")

if __name__ == "__main__":
    main()
