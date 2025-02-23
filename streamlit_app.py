import streamlit as st
import requests
import json

st.title('Customer Churn Prediction')

# Create form
with st.form("prediction_form"):
    st.write("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Required fields
        total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=500.0, value=100.0)
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=100.0, value=10.0)
        
        # Additional required fields
        state = st.selectbox("State", ["CA", "NY", "TX", "FL"])  # Add common state options
        account_length = st.number_input("Account Length", min_value=0, max_value=100, value=50)
        area_code = st.number_input("Area Code", min_value=0, max_value=999, value=415)
    
    with col2:
        # More required fields
        total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=100, value=3)
        total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=500.0, value=200.0)
        number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=0)
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        
        # Remaining required fields
        total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=100, value=50)
        total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=100, value=50)
        total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=500.0, value=100.0)
        total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=100, value=50)
        total_intl_charge = st.number_input("Total International Charge", min_value=0.0, max_value=50.0, value=5.0)
    
    submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare the input data matching ALL expected features
        input_data = {
            "State": state,
            "Account length": account_length,
            "Area code": area_code,
            "International plan": international_plan,
            "Voice mail plan": voice_mail_plan,
            "Number vmail messages": number_vmail_messages,
            "Total day minutes": total_day_minutes,
            "Total day calls": total_day_calls,
            "Total day charge": total_day_minutes * 0.01,  # Estimating charge
            "Total eve minutes": total_eve_minutes,
            "Total eve calls": total_eve_calls,
            "Total eve charge": total_eve_minutes * 0.01,  # Estimating charge
            "Total night minutes": total_night_minutes,
            "Total night calls": total_night_calls,
            "Total night charge": total_night_minutes * 0.01,  # Estimating charge
            "Total intl minutes": total_intl_minutes,
            "Total intl calls": total_intl_calls,
            "Total intl charge": total_intl_charge,
            "Customer service calls": customer_service_calls
        }
        
        try:
            # Make prediction request
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
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
