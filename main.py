import argparse
import mlflow
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# File paths for data
train_file = "churn-bigml-80.csv"
test_file = "churn-bigml-20.csv"

def run_full_pipeline():
    """Execute the complete ML pipeline."""
    with mlflow.start_run() as run:
        print("Running full pipeline...")
        
        mlflow.set_experiment("churn_prediction")
        mlflow.log_param("train_file", train_file)
        mlflow.log_param("test_file", test_file)
        
        print("\n🔹 Preparing data...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        
        print("\n🔹 Training model...")
        model = train_model(X_train, y_train)
        
        print("\n🔹 Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\n🔹 Saving model...")
        save_model(model)
        
        print("\n🔹 Loading and re-evaluating model...")
        loaded_model = load_model()
        evaluate_model(loaded_model, X_test, y_test)

def main():
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
    
    args = parser.parse_args()
    
    mlflow.set_experiment("churn_prediction")
    
    with mlflow.start_run() as run:
        if args.action == "prepare_data":
            print("\n🔹 Preparing data...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            
        elif args.action == "train_model":
            print("\n🔹 Training model...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train)
            
        elif args.action == "evaluate_model":
            print("\n🔹 Evaluating model...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            
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
            run_full_pipeline()
            
        else:
            print(
                "\n❌ Invalid action! Choose from: prepare_data, train_model, "
                "evaluate_model, save_model, load_model, or leave blank to run all."
            )

if __name__ == "__main__":
    main()
