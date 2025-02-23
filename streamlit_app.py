import streamlit as st
import requests
import json

st.title('Customer Churn Prediction')

# Create form
with st.form("prediction_form"):
    st.write("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=500.0, value=100.0)
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=100.0, value=10.0)
    
    with col2:
        total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=100, value=3)
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=500.0, value=200.0)
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=0)
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare the input data EXACTLY matching the expected input
        input_data = {
            "Total_day_minutes": total_day_minutes,
            "Customer_service_calls": customer_service_calls,
            "International_plan": international_plan,
            "Total_intl_minutes": total_intl_minutes,
            "Total_intl_calls": total_intl_calls,
            "Total_eve_minutes": total_eve_minutes,
            "Number_vmail_messages": number_vmail_messages,
            "Voice_mail_plan": voice_mail_plan
        }
        
        try:
            # Make prediction request using localhost for Docker container
            response = requests.post("http://localhost:8000/predict", json=input_data)
            response.raise_for_status()
            
            result = response.json()
            
            # Display prediction
            st.subheader("Prediction Results")
            st.write(f"Churn Prediction: {result['churn_prediction']}")
            st.write(f"Churn Probability: {result['churn_probability']:.2%}")
            
            # Add visual indicator
            if result['churn_prediction'] == "Yes":
                st.error("⚠️ High risk of churn!")
            else:
                st.success("✅ Low risk of churn")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error making prediction: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    st.error(f"Server response: {e.response.json()}")
                except:
                    st.error(f"Server response: {e.response.text}")
            st.error("Please check if the API server is running and all inputs are valid.")

# Add some custom styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("Customer Churn Prediction Model © 2024")
