import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.express as px

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered", initial_sidebar_state="auto")

# Custom CSS for styling (after set_page_config)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .stSubheader {
        color: #34495e;
        font-size: 24px;
        margin-top: 10px;
    }
    .stText {
        color: #7f8c8d;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and branding
st.markdown('<div class="stHeader">Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="stText">Predict customer churn with an interactive, user-friendly interface.</div>', unsafe_allow_html=True)

# Load model from artifacts (for fallback, if FastAPI fails)
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

# Input form for predictions with styling
st.markdown('<div class="stSubheader">Enter Customer Data</div>', unsafe_allow_html=True)
with st.form(key="churn_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=120.50, format="%.2f", help="Daily minutes of usage")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=5, help="Number of customer service interactions")
        international_plan = st.selectbox("International Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Does the customer have an international plan?")
    
    with col2:
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.20, format="%.2f", help="International minutes of usage")
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=5, help="Number of international calls")
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.00, format="%.2f", help="Evening minutes of usage")
    
    col3, col4 = st.columns(2)
    
    with col3:
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=1, help="Number of voicemail messages left")
        voice_mail_plan = st.selectbox("Voice Mail Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Does the customer have a voicemail plan?")
    
    submit_button = st.form_submit_button(label="Predict Churn", help="Click to predict churn probability")

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
            st.success(f"Churn Probability: {prediction:.4f}")
            st.markdown('<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>', unsafe_allow_html=True)
            
            # Visualize prediction with a gauge chart
            fig = px.pie(values=[prediction, 1 - prediction], names=["Churn Probability", "No Churn"],
                         title="Churn Probability Visualization", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error making prediction: {str(e)}. Please check if the API server is running and all inputs are valid.")
            
            # Fallback: Use local model if FastAPI fails
            if model:
                input_df = pd.DataFrame(input_data)
                prediction = model.predict_proba(input_df)[:, 1][0]
                st.success(f"Churn Probability (Local Model Fallback): {prediction:.4f}")
                st.markdown('<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>', unsafe_allow_html=True)
                
                # Visualize prediction with a gauge chart
                fig = px.pie(values=[prediction, 1 - prediction], names=["Churn Probability", "No Churn"],
                             title="Churn Probability Visualization (Fallback)", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

# Add footer for branding or additional info
st.markdown('<div class="stText" style="text-align: center; margin-top: 20px;">Powered by Streamlit & FastAPI | © 2025</div>', unsafe_allow_html=True)
