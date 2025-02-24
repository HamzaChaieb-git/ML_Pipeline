import streamlit as st
import pandas as pd
import joblib
import mlflow
import os
import requests
from datetime import datetime
import json

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

st.title("Churn Prediction Dashboard")

# Load model from artifacts (for simplicity, use the latest timestamped model)
def load_latest_model():
    models_dir = os.path.join("artifacts", "models")
    if not os.path.exists(models_dir):
        st.error("No models found in artifacts/models directory.")
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_v") and f.endswith(".joblib")]
    if not model_files:
        st.error("No model files found in artifacts/models.")
        return None
    
    latest_model = max(model_files, key=lambda x: x.split("v")[1].split(".joblib")[0])
    model_path = os.path.join(models_dir, latest_model)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Load model for local predictions (optional, for fallback)
model = load_latest_model()

if model:
    st.header("Model Overview")
    st.write("This dashboard visualizes the churn prediction model trained with XGBoost.")

    # Display the provided metrics directly (since you provided them)
    st.subheader("Model Metrics")
    metrics = {
        "train_samples": 2666,
        "test_samples": 667,
        "validation_0-error": 0.02156586966713549,
        "validation_0-logloss": 0.07141676126833896,
        "validation_0-auc": 0.9975934740679135,
        "validation_1-error": 0.0450281425891182,
        "validation_1-logloss": 0.16879593905897577,
        "validation_1-auc": 0.9258382642998028,
        "feature_importance_Total intl minutes": 0.05858517065644264,
        "feature_importance_Total eve minutes": 0.07122763246297836,
        "feature_importance_Number vmail messages": 0.07667464762926102,
        "feature_importance_Total intl calls": 0.08753249794244766,
        "feature_importance_Total day minutes": 0.10732730478048325,
        "feature_importance_Voice mail plan": 0.16968406736850739,
        "feature_importance_Customer service calls": 0.17224398255348206,
        "feature_importance_International plan": 0.2567247450351715,
        "train_accuracy": 0.9784341303328645,
        "train_logloss": 0.0714167611308755,
        "accuracy": 0.9565217391304348,
        "precision": 0.9230769230769231,
        "recall": 0.7578947368421053,
        "f1": 0.8323699421965318,
        "roc_auc": 0.9110783952889216,
        "log_loss": 0.1729857227185728,
        "precision_class_0": 0.9609507640067911,
        "recall_class_0": 0.9895104895104895,
        "f1_class_0": 0.9750215331610681,
        "precision_class_1": 0.9230769230769231,
        "recall_class_1": 0.7578947368421053,
        "f1_class_1": 0.8323699421965318,
        "true_negatives": 566,
        "false_positives": 6,
        "false_negatives": 23,
        "true_positives": 72,
        "balanced_accuracy": 0.8737026131762974,
        "precision_recall_ratio": 1.217948717948718,
        "false_positive_rate": 0.01048951048951049,
        "false_negative_rate": 0.24210526315789474,
        "positive_predictive_value": 0.9230769230769231,
        "negative_predictive_value": 0.9609507640067911
    }
    st.write(metrics)

    # Input form for predictions
    st.header("Predict Churn Probability")
    with st.form(key="churn_form"):
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=120.5, format="%.2f")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=3)
        international_plan = st.selectbox("International Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.2, format="%.2f")
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=5)
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.0, format="%.2f")
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=0)
        voice_mail_plan = st.selectbox("Voice Mail Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        input_data = {
            "Total day minutes": [total_day_minutes],
            "Customer service calls": [customer_service_calls],
            "International plan": [international_plan],
            "Total intl minutes": [total_intl_minutes],
            "Total intl calls": [total_intl_calls],
            "Total eve minutes": [total_eve_minutes],
            "Number vmail messages": [number_vmail_messages],
            "Voice mail plan": [voice_mail_plan]
        }

        try:
            # Try to make a prediction via FastAPI
            response = requests.post("http://localhost:8000/predict", json=input_data, timeout=10)
            response.raise_for_status()
            prediction = response.json()["churn_probabilities"][0]
            st.write(f"Churn Probability: {prediction:.4f}")
            st.write("Note: A higher probability indicates a higher likelihood of churn.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making prediction: {str(e)}. Please check if the API server is running and all inputs are valid.")
            
            # Fallback: Use local model if FastAPI fails
            if model:
                input_df = pd.DataFrame(input_data)
                prediction = model.predict_proba(input_df)[:, 1][0]
                st.write(f"Churn Probability (Local Model Fallback): {prediction:.4f}")
                st.write("Note: A higher probability indicates a higher likelihood of churn.")

    # Visualize feature importance (from artifacts)
    def load_feature_importance():
        training_dir = os.path.join("artifacts", "training")
        if not os.path.exists(training_dir):
            return None
        
        for folder in os.listdir(training_dir):
            feature_path = os.path.join(training_dir, folder, "feature_importance.csv")
            if os.path.exists(feature_path):
                return pd.read_csv(feature_path)
        return None

    feature_importance = load_feature_importance()
    if feature_importance is not None:
        st.header("Feature Importance")
        st.bar_chart(feature_importance.set_index('feature'))
