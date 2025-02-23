import streamlit as st
import requests
import json

st.title('Customer Churn Prediction')

# Create form
with st.form("prediction_form"):
    st.write("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Prepare the input data
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges
        }

        try:
            # Make prediction request
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

# Add some custom styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
