import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration for a wide, professional layout
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a premium, professional light-themed design with animations
st.markdown("""
    <style>
    /* Premium light background with subtle gradient */
    body, .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%) !important;
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: auto;
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 40px;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Header styling with bold blue text */
    .stHeader {
        color: #2c3e50;
        font-size: 56px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 0 2px 4px rgba(44, 62, 80, 0.1);
        font-family: 'Arial', sans-serif;
    }

    /* Subheader styling */
    .stSubheader {
        color: #34495e;
        font-size: 28px;
        margin-top: 25px;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }

    /* Text styling */
    .stText {
        color: #7f8c8d;
        font-size: 18px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }

    /* Input cards with white background, black text, and blue accents */
    .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        color: #333;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        transition: box-shadow 0.3s ease;
    }
    .stNumberInput:hover, .stSelectbox:hover {
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .stNumberInput > div > input, .stSelectbox > div > select {
        background-color: transparent !important;
        color: #333 !important;
        font-size: 18px;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .stNumberInput > div > input:focus, .stSelectbox > div > select:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 10px #3498db;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    /* Style for +/- buttons to match your screenshot */
    .stNumberInput > div > div > button {
        background-color: #3498db;
        color: #ffffff;
        border: none;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        font-size: 16px;
        margin: 0 8px;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.2);
        transition: background-color 0.3s, transform 0.2s;
    }
    .stNumberInput > div > div > button:hover {
        background-color: #2980b9;
        transform: scale(1.1);
    }

    /* Selectbox dropdown styling */
    .stSelectbox > div > select {
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        background: url('data:image/svg+xml;utf8,<svg fill="%23333" height="24" viewBox="0 0 24 24" width="24" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/></svg>') no-repeat right 12px center;
        padding-right: 40px;
        color: #333;
    }

    /* Button styling with premium blue theme */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 12px;
        padding: 16px 36px;
        font-size: 20px;
        border: none;
        margin-top: 25px;
        width: 100%;
        box-shadow: 0 6px 15px rgba(52, 152, 219, 0.3);
        font-family: 'Arial', sans-serif;
        transition: background-color 0.3s, box-shadow 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 8px 20px rgba(52, 152, 219, 0.5);
        transform: scale(1.05);
    }

    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #2d6a4f;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.1);
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #a4161a;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 10px rgba(164, 22, 26, 0.1);
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        animation: slideIn 0.5s ease-out;
    }

    /* Chart styling for gauge */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and branding with animation
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

# Input form for predictions with premium styling
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
    
    submit_button = st.form_submit_button(label="🔮 Predict", use_container_width=True)

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
            
            # Premium KPI-style gauge chart matching your screenshot
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction * 100,
                title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#2c3e50'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#7f8c8d', 'tickwidth': 1, 'tickfont': {'size': 14, 'color': '#7f8c8d'}},
                       'bar': {'color': "#4682b4"},  # Blue bar
                       'steps': [
                           {'range': [0, 50], 'color': "#f5f7fa"},
                           {'range': [50, 100], 'color': "#e9ecef"}],
                       'threshold': {'line': {'color': "#e74c3c", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                number={'valueformat': ".1f", 'font': {'size': 40, 'color': '#2c3e50'}}
            ))
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=400,
                width=600,
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
                
                # Premium KPI-style gauge chart for fallback
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#2c3e50'}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': '#7f8c8d', 'tickwidth': 1, 'tickfont': {'size': 14, 'color': '#7f8c8d'}},
                           'bar': {'color': "#4682b4"},
                           'steps': [
                               {'range': [0, 50], 'color': "#f5f7fa"},
                               {'range': [50, 100], 'color': "#e9ecef"}],
                           'threshold': {'line': {'color': "#e74c3c", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                    number={'valueformat': ".1f", 'font': {'size': 40, 'color': '#2c3e50'}}
                ))
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    height=400,
                    width=600,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
