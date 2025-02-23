import mlflow
import os
import sys

# Determine the base directory based on the environment
if os.path.exists('/app'):
    # Running in Docker container
    BASE_DIR = '/app'
else:
    # Running locally
    BASE_DIR = os.getcwd()  # Use current working directory

# Set MLflow tracking URI (adjust based on your setup)
# Use localhost:5001 if MLflow server is on Jenkins host with --network host
mlflow.set_tracking_uri("http://localhost:5001")  # Or "http://172.25.175.49:5001" if needed

def prepare_data():
    print("🔹 Preparing data...")
    output_file = os.path.join(BASE_DIR, 'data_output.txt')
    with open(output_file, 'w') as f:
        f.write("Data preparation completed\n")
    with mlflow.start_run(run_name="Data Preparation"):
        mlflow.log_param("step", "prepare_data")
        mlflow.log_artifact(output_file)
    print(f"🔹 Data preparation complete. File saved at: {output_file}")

def train_model():
    print("🔹 Training model...")
    output_file = os.path.join(BASE_DIR, 'train_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model training completed\nTraining XGBoost model...\n")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_param("step", "train_model")
        mlflow.log_metric("example_metric", 0.95)  # Replace with actual metrics
        mlflow.log_artifact(output_file)
    print(f"🔹 Model training complete. File saved at: {output_file}")

def evaluate_model():
    print("🔹 Evaluating model...")
    output_file = os.path.join(BASE_DIR, 'model_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model evaluation completed\nModel Accuracy: 0.9505\n")
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_param("step", "evaluate_model")
        mlflow.log_metric("example_score", 0.90)  # Replace with actual metrics
        mlflow.log_artifact(output_file)
    print(f"🔹 Evaluation complete. File saved at: {output_file}")

def save_model():
    print("🔹 Saving model...")
    output_file = os.path.join(BASE_DIR, 'model.pkl')
    with open(output_file, 'w') as f:
        f.write("dummy model")
    with mlflow.start_run(run_name="Model Saving"):
        mlflow.log_artifact(output_file)
    print(f"Model saved as model.pkl at: {output_file}")

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "prepare_data"
    if action == "prepare_data":
        prepare_data()
    elif action == "train_model":
        train_model()
    elif action == "evaluate_model":
        evaluate_model()
    elif action == "save_model":
        save_model()
    else:
        print(f"Unknown action: {action}. Use 'prepare_data', 'train_model', 'evaluate_model', or 'save_model'.")
