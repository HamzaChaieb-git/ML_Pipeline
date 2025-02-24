import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration as the first Streamlit command for a clean, centered layout
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for a simple, professional dark-themed design
st.markdown("""
    <style>
    /* Full-page dark background */
    body, .stApp {
        background-color: #1a1a2e !important;
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: auto;
    }
    .main {
        padding: 20px;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        background-color: transparent !important;
    }

    /* Header styling with red text */
    .stHeader {
        color: #ff4040;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: none;
    }

    /* Subheader styling */
    .stSubheader {
        color: #ffffff;
        font-size: 24px;
        margin-top: 15px;
        text-align: center;
    }

    /* Text styling */
    .stText {
        color: #a2a2a2;
        font-size: 16px;
        text-align: center;
    }

    /* Input fields and select boxes with dark blue theme */
    .stNumberInput, .stSelectbox {
        background-color: #16213e;
        border: 2px solid #0f3460;
        border-radius: 8px;
        padding: 10px;
        color: #ffffff;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stNumberInput > div > input, .stSelectbox > div > select {
        background-color: transparent !important;
        color: #ffffff !important;
        font-size: 16px;
    }
    .stNumberInput > div > input:focus, .stSelectbox > div > select:focus {
        outline: none;
        border-color: #ff4040;
        box-shadow: 0 0 5px #ff4040;
    }

    /* Button styling with simple hover */
    .stButton>button {
        background-color: #ff4040;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        border: none;
        margin-top: 15px;
        width: 100%;
        box-shadow: 0 2px 4px rgba(255, 64, 64, 0.2);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        box-shadow: 0 4px 8px rgba(255, 64, 64, 0.4);
    }

    /* Success message styling */
    .stSuccess {
        background-color: #2d6a4f;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(45, 106, 79, 0.2);
    }

    /* Error message styling */
    .stError {
        background-color: #a4161a;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(164, 22, 26, 0.2);
    }

    /* Chart styling for KPI-style gauge */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and branding
st.markdown('<div class="stHeader">Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="stText">Predict customer churn with a sleek, interactive, user-friendly interface.</div>', unsafe_allow_html=True)

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

# Input form for predictions with simple, professional styling
st.markdown('<div class="stSubheader">Enter Customer Data</div>', unsafe_allow_html=True)
with st.form(key="churn_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=120.50, format="%.2f", key="day_minutes")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1, key="service_calls")
        international_plan = st.selectbox("International Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="intl_plan")
    
    with col2:
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.20, format="%.2f", key="intl_minutes")
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=5, key="intl_calls")
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.00, format="%.2f", key="eve_minutes")
    
    col3, col4 = st.columns(2)
    
    with col3:
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=1, key="vmail_messages")
        voice_mail_plan = st.selectbox("Voice Mail Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="voice_plan")
    
    submit_button = st.form_submit_button(label="Predict Churn")

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
            st.markdown(f'<div class="stSuccess">Churn Probability: {prediction:.4f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>', unsafe_allow_html=True)
            
            # KPI-style gauge chart matching your screenshot
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction * 100,
                title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#ffffff'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#ffffff', 'tickwidth': 1, 'tickfont': {'size': 12, 'color': '#ffffff'}},
                       'bar': {'color': "#4682b4"},  # Dark blue bar
                       'steps': [
                           {'range': [0, 50], 'color': "#16213e"},
                           {'range': [50, 100], 'color': "#0f3460"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                number={'valueformat': ".1f", 'font': {'size': 36, 'color': '#ffffff'}}
            ))
            fig.update_layout(
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                height=300,
                width=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        except requests.exceptions.RequestException as e:
            st.markdown(f'<div class="stError">Error making prediction: {str(e)}. Please check if the API server is running and all inputs are valid.</div>', unsafe_allow_html=True)
            
            # Fallback: Use local model if FastAPI fails
            if model:
                input_df = pd.DataFrame(input_data)
                prediction = model.predict_proba(input_df)[:, 1][0]
                st.markdown(f'<div class="stSuccess">Churn Probability (Local Model Fallback): {prediction:.4f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>', unsafe_allow_html=True)
                
                # KPI-style gauge chart for fallback
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#ffffff'}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': '#ffffff', 'tickwidth': 1, 'tickfont': {'size': 12, 'color': '#ffffff'}},
                           'bar': {'color': "#4682b4"},  # Dark blue bar
                           'steps': [
                               {'range': [0, 50], 'color': "#16213e"},
                               {'range': [50, 100], 'color': "#0f3460"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                    number={'valueformat': ".1f", 'font': {'size': 36, 'color': '#ffffff'}}
                ))
                fig.update_layout(
                    paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#1a1a2e",
                    height=300,
                    width=500,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
