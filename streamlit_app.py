import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import requests
import plotly.graph_objects as go
from datetime import datetime

# Try importing sklearn with error handling
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
except ImportError:
    st.error("Please install scikit-learn: pip install scikit-learn")
    st.stop()

MODELS_DIR = "models"

@st.cache_data
def load_sample_data():
    """Load sample data for testing"""
    try:
        df = pd.read_csv('data/cleaned_test (6).csv')
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all trained models from the models directory"""
    models = {}
    try:
        # Try to load each model
        model_files = os.listdir(MODELS_DIR)
        for model_file in model_files:
            if model_file.endswith('_model.joblib'):
                model_name = model_file.replace('_model.joblib', '')
                try:
                    model_path = os.path.join(MODELS_DIR, model_file)
                    models[model_name] = joblib.load(model_path)
                except Exception as e:
                    st.error(f"Error loading {model_name}: {str(e)}")
        
        return models if models else None
        
    except Exception as e:
        st.error(f"General error: {str(e)}")
        return None

def create_feature_input(feature, feature_type, sample_value=None):
    """Create appropriate input widget based on feature type"""
    if feature in ['International plan', 'Voice mail plan']:
        default_value = 'Yes' if sample_value == 1 else 'No' if sample_value == 0 else 'No'
        return st.selectbox(feature, ['No', 'Yes'], index=0 if default_value == 'No' else 1)
    else:
        default_value = float(sample_value) if sample_value is not None else 0.0
        return st.number_input(
            feature,
            value=default_value,
            format="%.2f"  # Simplified to two decimal places for consistency with screenshot
        )

def process_input_data(input_data):
    """Process input data and convert to correct types"""
    processed_data = input_data.copy()
    
    # Convert Yes/No to 1/0 for categorical variables
    for key in ['International plan', 'Voice mail plan']:
        if key in processed_data:
            processed_data[key] = 1 if processed_data[key] == 'Yes' else 0
            processed_data[key] = np.float32(processed_data[key])
    
    # Convert all other values to float32
    for key, value in processed_data.items():
        if key not in ['International plan', 'Voice mail plan']:
            processed_data[key] = np.float32(value)
    
    return processed_data

def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Prediction",
        page_icon="📱",
        layout="centered"
    )
    
    st.markdown('<div style="text-align: center;"><h1 style="color: #4a90e2; font-family: Arial, sans-serif; font-size: 48px; margin-bottom: 10px;">Churn Prediction Dashboard</h1></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; font-family: Arial, sans-serif; font-size: 16px; margin-bottom: 20px;">Predict customer churn with an interactive, user-friendly interface.</div>', unsafe_allow_html=True)
    
    try:
        # Load models and sample data
        models = load_models()
        sample_data = load_sample_data()
        
        if not models:
            st.error("Failed to load necessary model files.")
            return
        
        # Sidebar - Model selection and sample data
        with st.sidebar:
            st.markdown('<h3 style="color: #333; font-family: Arial, sans-serif; font-size: 24px;">🔧 Model Settings</h3>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                'Choose a model',
                options=list(models.keys()),
                help='Select the machine learning model to use for prediction'
            )
            
            st.markdown('<h3 style="color: #333; font-family: Arial, sans-serif; font-size: 24px;">📊 Sample Data Selection</h3>', unsafe_allow_html=True)
            if sample_data is not None:
                use_sample = st.checkbox('Use sample data for testing')
                if use_sample:
                    # Show sample data in a table with pagination
                    st.dataframe(sample_data, height=200)
                    selected_index = st.number_input(
                        'Select row number to test',
                        min_value=0,
                        max_value=len(sample_data)-1,
                        value=0
                    )
                    selected_sample = sample_data.iloc[selected_index]
                    
                    # Show actual churn value for selected sample
                    actual_churn = selected_sample['Churn']
                    st.markdown(f'<p style="color: #666; font-family: Arial, sans-serif; font-size: 16px;">Actual Churn Value: {"Yes" if actual_churn else "No"}</p>', unsafe_allow_html=True)
                else:
                    selected_sample = None
            else:
                selected_sample = None
                st.error("Sample data not available")
        
        # Main panel - Input features
        st.markdown('<div style="text-align: center;"><h2 style="color: #333; font-family: Arial, sans-serif; font-size: 24px;">📝 Enter Customer Information</h2></div>', unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Get features from sample data
        features = sample_data.columns.drop('Churn') if sample_data is not None else []
        
        # Create input fields
        input_data = {}
        for i, feature in enumerate(features):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                sample_value = selected_sample[feature] if selected_sample is not None else None
                input_data[feature] = create_feature_input(
                    feature,
                    None,  # feature type not needed with scaled data
                    sample_value
                )
        
        # Make prediction
        if st.button('🔮 Predict', use_container_width=True):
            try:
                with st.spinner('Analyzing customer data...'):
                    # Process the input data
                    processed_data = process_input_data(input_data)
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame([processed_data])
                    
                    # Get prediction
                    model = models[selected_model]
                    prediction = model.predict(input_df)
                    probability = model.predict_proba(input_df)
                    
                    # Show results
                    st.markdown('<div style="text-align: center;"><h2 style="color: #333; font-family: Arial, sans-serif; font-size: 24px;">🎯 Prediction Results</h2></div>', unsafe_allow_html=True)
                    
                    # Create columns for the results
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if prediction[0]:
                            st.markdown('<div style="background-color: #f2dede; color: #a4161a; padding: 12px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif; font-size: 18px;">⚠️ Customer is likely to churn!</div>', unsafe_allow_html=True)
                            st.markdown(f'<p style="color: #666; font-family: Arial, sans-serif; font-size: 16px;">Churn probability: {probability[0][1]:.1%}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: #dff0d8; color: #2d6a4f; padding: 12px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif; font-size: 18px;">✅ Customer is likely to stay!</div>', unsafe_allow_html=True)
                            st.markdown(f'<p style="color: #666; font-family: Arial, sans-serif; font-size: 16px;">Retention probability: {probability[0][0]:.1%}</p>', unsafe_allow_html=True)
                    
                    with result_col2:
                        # Show confidence and gauge
                        if prediction[0]:
                            churn_prob = probability[0][1] * 100
                        else:
                            churn_prob = probability[0][0] * 100
                        
                        st.markdown(f'<p style="color: #666; font-family: Arial, sans-serif; font-size: 16px;">Confidence: {churn_prob:.1f}%</p>', unsafe_allow_html=True)
                        st.progress(churn_prob/100)
                        
                        # KPI-style gauge chart matching your screenshot
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_prob if prediction[0] else 100 - churn_prob,
                            title={'text': "Churn Probability (%)" if prediction[0] else "Retention Probability (%)", 'font': {'size': 16, 'color': '#333'}},
                            gauge={'axis': {'range': [0, 100], 'tickcolor': '#666', 'tickwidth': 1, 'tickfont': {'size': 12, 'color': '#666'}},
                                'bar': {'color': "#4682b4"},  # Blue bar
                                'steps': [
                                    {'range': [0, 50], 'color': "#e0e5ec"},
                                    {'range': [50, 100], 'color': "#d0d6e0"}],
                                'threshold': {'line': {'color': "#ff4040", 'width': 4}, 'thickness': 0.75, 'value': 80}},
                            number={'valueformat': ".1f", 'font': {'size': 36, 'color': '#333'}}
                        ))
                        fig.update_layout(
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            height=300,
                            width=500,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == '__main__':
    main()
