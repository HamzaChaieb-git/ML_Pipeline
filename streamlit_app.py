import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration for a simple, centered layout
st.set_page_config(
    page_title="Churn Prediction Dashboard", page_icon="📊", layout="centered"
)

# Custom CSS for a simple, professional light-themed design
st.markdown(
    """
    <style>
    /* Clean light background */
    body, .stApp {
        background-color: #f5f7fa;
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: auto;
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 20px;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Header styling with blue text */
    .stHeader {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        font-family: 'Arial', sans-serif;
    }

    /* Subheader styling */
    .stSubheader {
        color: #34495e;
        font-size: 20px;
        margin-top: 15px;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }

    /* Text styling */
    .stText {
        color: #7f8c8d;
        font-size: 16px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        line-height: 1.5;
    }

    /* Input fields with clean design */
    .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        color: #333;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }
    .stNumberInput > div > input, .stSelectbox > div > select {
        background-color: transparent !important;
        color: #333 !important;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .stNumberInput > div > input:focus, .stSelectbox > div > select:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 5px #3498db;
        transition: border-color 0.3s, box-shadow 0.3s;
    }

    /* Button styling with simple blue theme */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        margin-top: 15px;
        width: 100%;
        box-shadow: 0 2px 5px rgba(52, 152, 219, 0.2);
        font-family: 'Arial', sans-serif;
        transition: background-color 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }

    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #2d6a4f;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }

    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #a4161a;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }

    /* Chart styling for gauge */
    .stPlotlyChart {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
        margin-bottom: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Simple title and branding
st.markdown(
    '<div class="stHeader">Churn Prediction Dashboard</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="stText">Predict customer churn with a simple, user-friendly interface.</div>',
    unsafe_allow_html=True,
)


# Load model from artifacts (for fallback, if FastAPI fails)
def load_latest_model():
    models_dir = os.path.join("artifacts", "models")
    if not os.path.exists(models_dir):
        st.error("No models found in artifacts/models directory.")
        return None

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".joblib")
    ]
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

# Input form for predictions with simple styling
st.markdown(
    '<div class="stSubheader">Enter Customer Data</div>', unsafe_allow_html=True
)
with st.form(key="churn_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        total_day_minutes = st.number_input(
            "Total Day Minutes",
            min_value=0.0,
            value=120.50,
            format="%.2f",
            key="day_minutes",
        )
        customer_service_calls = st.number_input(
            "Customer Service Calls", min_value=0, value=1, key="service_calls"
        )
        international_plan = st.selectbox(
            "International Plan",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="intl_plan",
        )

    with col2:
        total_intl_minutes = st.number_input(
            "Total International Minutes",
            min_value=0.0,
            value=10.20,
            format="%.2f",
            key="intl_minutes",
        )
        total_intl_calls = st.number_input(
            "Total International Calls", min_value=0, value=5, key="intl_calls"
        )
        total_eve_minutes = st.number_input(
            "Total Evening Minutes",
            min_value=0.0,
            value=200.00,
            format="%.2f",
            key="eve_minutes",
        )

    col3, col4 = st.columns(2)

    with col3:
        number_vmail_messages = st.number_input(
            "Number of Voicemail Messages", min_value=0, value=1, key="vmail_messages"
        )
        voice_mail_plan = st.selectbox(
            "Voice Mail Plan",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="voice_plan",
        )

    submit_button = st.form_submit_button(label="Predict", use_container_width=True)

    if submit_button:
        input_data = {
            "Total day minutes": [total_day_minutes],
            "Customer service calls": [customer_service_calls],
            "International plan": [international_plan],
            "Total intl minutes": [total_intl_minutes],
            "Total intl calls": [total_intl_calls],
            "Total eve minutes": [total_eve_minutes],
            "Number vmail messages": [number_vmail_messages],
            "Voice mail plan": [voice_mail_plan],
        }

        try:
            # Use the FastAPI service name for Docker network communication
            response = requests.post(
                "http://fastapi:8000/predict", json=input_data, timeout=10
            )
            response.raise_for_status()
            prediction = response.json()["churn_probabilities"][0]
            st.markdown(
                f'<div class="stSuccess">Churn Probability: {prediction:.4f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>',
                unsafe_allow_html=True,
            )

            # Simple KPI-style gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={
                        "text": "Churn Probability (%)",
                        "font": {"size": 14, "color": "#2c3e50"},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#7f8c8d",
                            "tickwidth": 1,
                            "tickfont": {"size": 12, "color": "#7f8c8d"},
                        },
                        "bar": {"color": "#3498db"},
                        "steps": [
                            {"range": [0, 50], "color": "#f5f7fa"},
                            {"range": [50, 100], "color": "#e9ecef"},
                        ],
                        "threshold": {
                            "line": {"color": "#e74c3c", "width": 3},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                    number={
                        "valueformat": ".1f",
                        "font": {"size": 24, "color": "#2c3e50"},
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=250,
                width=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.markdown(
                f'<div class="stError">Error making prediction: {str(e)}. Please check if the FastAPI server is running at fastapi:8000.</div>',
                unsafe_allow_html=True,
            )

            # Fallback: Use local model if FastAPI fails
            if model:
                input_df = pd.DataFrame(input_data)
                prediction = model.predict_proba(input_df)[:, 1][0]
                st.markdown(
                    f'<div class="stSuccess">Churn Probability (Local Model Fallback): {prediction:.4f}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="stText">Note: A higher probability indicates a higher likelihood of churn.</div>',
                    unsafe_allow_html=True,
                )

                # Simple KPI-style gauge chart for fallback
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction * 100,
                        title={
                            "text": "Churn Probability (%)",
                            "font": {"size": 14, "color": "#2c3e50"},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickcolor": "#7f8c8d",
                                "tickwidth": 1,
                                "tickfont": {"size": 12, "color": "#7f8c8d"},
                            },
                            "bar": {"color": "#3498db"},
                            "steps": [
                                {"range": [0, 50], "color": "#f5f7fa"},
                                {"range": [50, 100], "color": "#e9ecef"},
                            ],
                            "threshold": {
                                "line": {"color": "#e74c3c", "width": 3},
                                "thickness": 0.75,
                                "value": 80,
                            },
                        },
                        number={
                            "valueformat": ".1f",
                            "font": {"size": 24, "color": "#2c3e50"},
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    height=250,
                    width=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
