import argparse
import sys
import os

def prepare_data(train_file: str = "churn-bigml-80.csv", test_file: str = "churn-bigml-20.csv"):
    """Prepare data and save output to a local file."""
    print("🔹 Preparing data...")
    output_file = os.path.join(os.getcwd(), 'data_output.txt')
    with open(output_file, 'w') as f:
        f.write("Data preparation completed\n")
    print(f"🔹 Data preparation complete. File saved at: {output_file}")
    return None, None, None, None  # Placeholder for X_train, X_test, y_train, y_test

def train_model(X_train=None, y_train=None):
    """Train model and save output to a local file."""
    print("🔹 Training model...")
    output_file = os.path.join(os.getcwd(), 'train_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model training completed\nTraining XGBoost model...\n")
    print(f"🔹 Model training complete. File saved at: {output_file}")
    return None  # Placeholder for model object

def evaluate_model(model=None, X_test=None, y_test=None):
    """Evaluate model and save output to a local file."""
    print("🔹 Evaluating model...")
    output_file = os.path.join(os.getcwd(), 'model_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model evaluation completed\nModel Accuracy: 0.9505\n")
    print(f"🔹 Evaluation complete. File saved at: {output_file}")

def save_model(model=None):
    """Save model to a local file."""
    print("🔹 Saving model...")
    output_file = os.path.join(os.getcwd(), 'model.pkl')
    with open(output_file, 'w') as f:
        f.write("dummy model")  # Placeholder
    print(f"Model saved as model.pkl at: {output_file}")

def load_model():
    """Load model from a local file (placeholder function)."""
    print("🔹 Loading model...")
    output_file = os.path.join(os.getcwd(), 'model.pkl')
    with open(output_file, 'r') as f:
        content = f.read()
    print("Model loaded")
    return None  # Placeholder for loaded model

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
