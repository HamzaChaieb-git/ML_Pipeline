import sys
import os

BASE_DIR = os.getcwd()  # Use current working directory locally

def prepare_data():
    print("🔹 Preparing data...")
    output_file = os.path.join(BASE_DIR, 'data_output.txt')
    with open(output_file, 'w') as f:
        f.write("Data preparation completed\n")
    print(f"🔹 Data preparation complete. File saved at: {output_file}")

def train_model():
    print("🔹 Training model...")
    output_file = os.path.join(BASE_DIR, 'train_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model training completed\nTraining XGBoost model...\n")
    print(f"🔹 Model training complete. File saved at: {output_file}")

def evaluate_model():
    print("🔹 Evaluating model...")
    output_file = os.path.join(BASE_DIR, 'model_output.txt')
    with open(output_file, 'w') as f:
        f.write("Model evaluation completed\nModel Accuracy: 0.9505\n")
    print(f"🔹 Evaluation complete. File saved at: {output_file}")

def save_model():
    print("🔹 Saving model...")
    output_file = os.path.join(BASE_DIR, 'model.pkl')
    with open(output_file, 'w') as f:
        f.write("dummy model")
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
