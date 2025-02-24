import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered", initial_sidebar_state="auto")

# Custom CSS for a premium, dark-themed design with animations
st.markdown("""
    <style>
    /* Main container styling with dark background and subtle animation */
    .main {
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Header styling */
    .stHeader {
        color: #e94560;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0 2px 4px rgba(233, 69, 96, 0.3);
    }

    /* Subheader styling */
    .stSubheader {
        color: #ffffff;
        font-size: 28px;
        margin-top: 15px;
        text-shadow: 0 1px 3px rgba(255, 255, 255, 0.2);
    }

    /* Text styling */
    .stText {
        color: #a2a2a2;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Input fields and select boxes */
    .stNumberInput, .stSelectbox {
        background-color: #16213e;
        border: 2px solid #0f3460;
        border-radius: 10px;
        padding: 10px;
        color: #ffffff;
    }
    .stNumberInput > div > input, .stSelectbox > div > select {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    .stNumberInput > div > input:focus, .stSelectbox > div > select:focus {
        outline: none;
        border-color: #e94560;
        box-shadow: 0 0 5px #e94560;
    }

    /* Button styling with hover animation */
    .stButton>button {
        background-color: #e94560;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 18px;
        border: none;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.5);
    }

    /* Success and error messages */
    .stSuccess {
        background-color: #2d6a4f;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    .stError {
        background-color: #a4161a;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Chart styling */
    .stPlotlyChart {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Footer styling */
    .stFooter {
        color: #a2a2a2;
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        opacity: 0.8;
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

# Input form for predictions with advanced styling and tooltips
st.markdown('<div class="stSubheader">Enter Customer Data</div>', unsafe_allow_html=True)
with st.form(key="churn_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=120.50, format="%.2f", 
                                          help="Daily minutes of usage (hover for details)", 
                                          key="day_minutes")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=10, 
                                               help="Number of customer service interactions (hover for details)", 
                                               key="service_calls")
        international_plan = st.selectbox("International Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                                        help="Does the customer have an international plan? (hover for details)", 
                                        key="intl_plan")
    
    with col2:
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.20, format="%.2f", 
                                           help="International minutes of usage (hover for details)", 
                                           key="intl_minutes")
        total_intl_calls = st.number_input("Total International Calls", min_value=0, value=5, 
                                         help="Number of international calls (hover for details)", 
                                         key="intl_calls")
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.00, format="%.2f", 
                                          help="Evening minutes of usage (hover for details)", 
                                          key="eve_minutes")
    
    col3, col4 = st.columns(2)
    
    with col3:
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=1, 
                                              help="Number of voicemail messages left (hover for details)", 
                                              key="vmail_messages")
        voice_mail_plan = st.selectbox("Voice Mail Plan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                                     help="Does the customer have a voicemail plan? (hover for details)", 
                                     key="voice_plan")
    
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
            st.markdown(f'<div class="stSuccess">Churn Probability: {prediction:.4f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>', unsafe_allow_html=True)
            
            # Advanced visualization: Animated gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction * 100,
                title={'text': "Churn Probability (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#e94560"},
                       'steps': [
                           {'range': [0, 50], 'color': "#16213e"},
                           {'range': [50, 100], 'color': "#0f3460"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                number={'valueformat': ".1f", 'font': {'size': 24, 'color': "#ffffff"}}
            ))
            fig.update_layout(
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                height=300,
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
                
                # Advanced visualization: Animated gauge chart for fallback
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={'text': "Churn Probability (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#e94560"},
                           'steps': [
                               {'range': [0, 50], 'color': "#16213e"},
                               {'range': [50, 100], 'color': "#0f3460"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                    number={'valueformat': ".1f", 'font': {'size': 24, 'color': "#ffffff"}}
                ))
                fig.update_layout(
                    paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#1a1a2e",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

# Add an interactive footer with branding and animation
st.markdown('<div class="stFooter">Powered by Streamlit & FastAPI | © 2025 <span style="color: #e94560;">✨</span></div>', unsafe_allow_html=True)

# Add a subtle animation on page load
st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        const elements = document.querySelectorAll('.stContainer');
        elements.forEach(element => {
            element.style.opacity = 0;
            setTimeout(() => {
                element.style.transition = 'opacity 1s ease-in';
                element.style.opacity = 1;
            }, 100);
        });
    });
    </script>
""", unsafe_allow_html=True)
