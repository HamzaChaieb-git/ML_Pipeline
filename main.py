import argparse
import mlflow
import mlflow.sklearn
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

def run_full_pipeline_with_mlflow():
    """Execute the complete ML pipeline with MLflow tracking."""
    
    # Set MLflow experiment
    mlflow.set_experiment("churn_prediction")
    
    with mlflow.start_run() as run:
        print("Running full pipeline with MLflow tracking...")
        
        # Log input dataset info
        mlflow.log_param("train_dataset", train_file)
        mlflow.log_param("test_dataset", test_file)
        
        print("\n🔹 Preparing data...")
        X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
        
        # Log dataset metrics
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        print("\n🔹 Training model...")
        model = train_model(X_train, y_train)
        
        # Log model parameters
        model_params = model.get_params()
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        print("\n🔹 Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        print("\n🔹 Saving model...")
        save_model(model)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("\n🔹 Loading and re-evaluating model...")
        loaded_model = load_model()
        evaluate_model(loaded_model, X_test, y_test)
        
        return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline Controller with MLflow")
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
    
    # Execute based on argument
    with mlflow.start_run() as run:
        if args.action == "prepare_data":
            print("\n🔹 Preparing data...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            mlflow.log_param("action", "prepare_data")
            mlflow.log_param("n_features", X_train.shape[1])
            
        elif args.action == "train_model":
            print("\n🔹 Training model...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train)
            mlflow.log_param("action", "train_model")
            mlflow.sklearn.log_model(model, "model")
            
        elif args.action == "evaluate_model":
            print("\n🔹 Evaluating model...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_param("action", "evaluate_model")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
        elif args.action == "save_model":
            print("\n🔹 Saving model...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            model = train_model(X_train, y_train)
            save_model(model)
            mlflow.log_param("action", "save_model")
            mlflow.sklearn.log_model(model, "saved_model")
            
        elif args.action == "load_model":
            print("\n🔹 Loading model and re-evaluating...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            loaded_model = load_model()
            metrics = evaluate_model(loaded_model, X_test, y_test)
            mlflow.log_param("action", "load_model")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
        elif args.action == "all":
            mlflow.log_param("action", "full_pipeline")
            run_full_pipeline_with_mlflow()
            
        else:
            print(
                "\n❌ Invalid action! Choose from: prepare_data, train_model, "
                "evaluate_model, save_model, load_model, or leave blank to run all."
            )

if __name__ == "__main__":
    main()
