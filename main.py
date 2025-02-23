import mlflow
import os

# Set MLflow tracking URI (use localhost if using --network host)
mlflow.set_tracking_uri("http://172.25.175.49:5001")  # Or "http://localhost:5001" if using --network host

def prepare_data():
    print("🔹 Preparing data...")
    with open('/app/data_output.txt', 'w') as f:
        f.write("Data preparation completed\n")
    with mlflow.start_run(run_name="Data Preparation"):
        mlflow.log_param("step", "prepare_data")
        mlflow.log_artifact("/app/data_output.txt")
    print("🔹 Data preparation complete")

def train_model():
    print("🔹 Training model...")
    with open('/app/train_output.txt', 'w') as f:
        f.write("Model training completed\n")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_param("step", "train_model")
        mlflow.log_metric("example_metric", 0.95)
        mlflow.log_artifact("/app/train_output.txt")
    print("🔹 Model training complete")

def evaluate_model():
    print("🔹 Evaluating model...")
    with open('/app/model_output.txt', 'w') as f:
        f.write("Model evaluation completed\nModel Accuracy: 0.9505\n")
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_param("step", "evaluate_model")
        mlflow.log_metric("example_score", 0.90)
        mlflow.log_artifact("/app/model_output.txt")
    print("🔹 Evaluation complete")

def save_model():
    print("🔹 Saving model...")
    with open('/app/model.pkl', 'w') as f:
        f.write("dummy model")
    with mlflow.start_run(run_name="Model Saving"):
        mlflow.log_artifact("/app/model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "prepare_data"
    if action == "prepare_data":
        prepare_data()
    elif action == "train_model":
        train_model()
    elif action == "evaluate_model":
        evaluate_model()
    elif action == "save_model":
        save_model()
