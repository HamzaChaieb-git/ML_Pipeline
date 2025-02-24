import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration for a simple, centered layout
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="centered"
)

# Custom CSS for a simple, professional light-themed design
st.markdown("""
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
        box-shadow: 0
