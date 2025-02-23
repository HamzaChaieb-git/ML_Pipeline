"""Main script to control and orchestrate the machine learning pipeline."""

import argparse
from data_processing import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model
from model_persistence import save_model, load_model
import mlflow


def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")

    print("\n🔹 Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)

    print("\n🔹 Training model...")
    model = train_model(X_train, y_train)

    print("\n🔹 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\n🔹 Saving model...")
    save_model(model)

    print("\n🔹 Loading and re-evaluating model...")
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
        help=(
            "Action to perform: prepare_data, train_model, evaluate_model, "
            "save_model, load_model, or run all steps by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the pipeline based on command-line arguments."""
    # Set MLflow tracking URI (using SQLite backend)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    args = parse_arguments()
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    if args.action == "prepare_data":
        print("\n🔹 Preparing data...")
        prepare_data(train_file, test_file)

    elif args.action == "train_model":
        print("\n🔹 Training model...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        train_model(X_train, y_train)

    elif args.action == "evaluate_model":
        print("\n🔹 Evaluating model...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    elif args.action == "save_model":
        print("\n🔹 Saving model...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        model = train_model(X_train, y_train)
        save_model(model)

    elif args.action == "load_model":
        print("\n🔹 Loading model and re-evaluating...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        loaded_model = load_model()
        evaluate_model(loaded_model, X_test, y_test)

    elif args.action == "all":
        run_full_pipeline(train_file, test_file)

    else:
        print(
            "\n❌ Invalid action! Choose from: prepare_data, train_model, "
            "evaluate_model, save_model, load_model, or leave blank to run all."
        )


if __name__ == "__main__":
    main()
