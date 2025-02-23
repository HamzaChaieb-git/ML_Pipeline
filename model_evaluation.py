import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def prepare_data(train_file: str = "churn-bigml-80.csv", test_file: str = "churn-bigml-20.csv"):
    """Prepare data by loading and splitting CSV files, returning train/test splits."""
    print("🔹 Preparing data...")
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print("Warning: CSV files not found, using placeholder data.")
        # Placeholder data with 'Churn' as target
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000),
            'Churn': np.random.randint(0, 2, 1000)
        })
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Check for 'Churn' column
    if 'Churn' not in train_df.columns or 'Churn' not in test_df.columns:
        raise ValueError("Missing 'Churn' column in data")

    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']

    print("🔹 Data preparation complete")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train an XGBoost model."""
    print("🔹 Training model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("🔹 Model training complete")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print classification metrics."""
    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("🔹 Evaluation complete")

def save_model(model, filename: str = "model.joblib"):
    """Save the model to a local file."""
    print("🔹 Saving model...")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename: str = "model.joblib"):
    """Load the model from a local file."""
    print("🔹 Loading model...")
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline."""
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
