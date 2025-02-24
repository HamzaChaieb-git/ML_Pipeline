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

    # Load evaluation metrics from MLflow or artifacts
    def load_metrics(run_id: str):
        client = mlflow.tracking.MlflowClient()
        try:
            metrics = client.get_run(run_id).data.metrics
            return metrics
        except Exception as e:
            st.warning(f"Could not load metrics from MLflow: {str(e)}")
            return None

    # Get the latest run ID from the churn_prediction experiment
    experiment = mlflow.get_experiment_by_name("churn_prediction")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            metrics = load_metrics(run_id)
            if metrics:
                st.subheader("Model Metrics")
                st.write(metrics)

    # Input form for predictions
    st.header("Predict Churn Probability")
    with st.form(key="churn_form"):
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=120.5)
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=3)
        international_plan = st.selectbox("International Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.2)
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=5)
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.0)
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

# Run the Streamlit app with explicit host and port
if __name__ == "__main__":
    st.run(server_address="0.0.0.0", server_port=8501)
