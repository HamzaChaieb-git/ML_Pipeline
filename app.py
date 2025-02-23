import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.joblib')
    except:
        st.error("⚠️ Model file not found. Please ensure model.joblib exists in the same directory.")
        return None

# Feature preprocessing
def preprocess_features(data):
    le = LabelEncoder()
    categorical_features = ['International plan', 'Voice mail plan']
    for feature in categorical_features:
        data[feature] = le.fit_transform(data[feature])
    return data

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Model Overview", "Single Prediction", "Batch Prediction"]
    )

    # Load model
    model = load_model()
    
    if model is None:
        st.stop()

    # Define features
    features = [
        'Total day minutes',
        'Customer service calls',
        'International plan',
        'Total intl minutes',
        'Total intl calls',
        'Total eve minutes',
        'Number vmail messages',
        'Voice mail plan'
    ]

    if page == "Model Overview":
        st.title("📊 Churn Prediction Model Overview")
        
        # Model Information
        st.header("Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Model Type", value="Random Forest")
        with col2:
            st.metric(label="Number of Features", value=len(features))
        with col3:
            st.metric(label="Feature Importance Available", value="Yes")

        # Feature Importance
        st.header("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Plot'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Single Prediction":
        st.title("🎯 Single Customer Churn Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=500.0, value=100.0)
            service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
            intl_plan = st.selectbox("International Plan", ["No", "Yes"])
            intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=100.0, value=10.0)

        with col2:
            intl_calls = st.number_input("Total International Calls", min_value=0, max_value=100, value=3)
            eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=500.0, value=200.0)
            vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=0)
            vmail_plan = st.selectbox("Voicemail Plan", ["No", "Yes"])

        if st.button("Predict Churn Probability"):
            # Create input dataframe
            input_data = pd.DataFrame([[
                day_minutes, service_calls, intl_plan, intl_minutes,
                intl_calls, eve_minutes, vmail_messages, vmail_plan
            ]], columns=features)
            
            # Preprocess
            processed_data = preprocess_features(input_data)
            
            # Make prediction
            probability = model.predict_proba(processed_data)[0][1]
            prediction = "High Risk" if probability >= 0.5 else "Low Risk"
            
            # Show prediction
            st.header("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Churn Risk",
                    value=prediction,
                    delta=f"{probability:.1%}"
                )
            
            with col2:
                # Create gauge chart for probability
                fig = px.pie(
                    values=[probability, 1-probability],
                    names=['Churn Risk', 'Retention Probability'],
                    hole=0.7,
                    title="Churn Probability"
                )
                st.plotly_chart(fig)

    else:  # Batch Prediction
        st.title("📑 Batch Churn Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Check if required features are present
            missing_features = [feat for feat in features if feat not in data.columns]
            
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                st.stop()
            
            # Process data
            input_data = data[features].copy()
            processed_data = preprocess_features(input_data)
            
            # Make predictions
            probabilities = model.predict_proba(processed_data)[:, 1]
            predictions = model.predict(processed_data)
            
            # Add predictions to dataframe
            results = data.copy()
            results['Churn Probability'] = probabilities
            results['Predicted Churn'] = predictions
            
            # Show results
            st.header("Prediction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Customers",
                    value=len(results)
                )
            
            with col2:
                high_risk = (results['Churn Probability'] >= 0.5).sum()
                st.metric(
                    label="High Risk Customers",
                    value=high_risk,
                    delta=f"{high_risk/len(results):.1%}"
                )
            
            with col3:
                avg_prob = results['Churn Probability'].mean()
                st.metric(
                    label="Average Churn Probability",
                    value=f"{avg_prob:.1%}"
                )
            
            # Distribution plot
            fig = px.histogram(
                results,
                x='Churn Probability',
                nbins=50,
                title='Distribution of Churn Probabilities'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show dataframe
            st.dataframe(results)
            
            # Download link
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
