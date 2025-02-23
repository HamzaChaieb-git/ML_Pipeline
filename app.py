from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = FastAPI()

# Initialize model as None
model = None

# Load the model when the app starts
def load_model():
    global model
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception(f"Failed to load model: {e}")

# Load model at startup
load_model()

class ChurnPredictionInput(BaseModel):
    Total_day_minutes: float
    Customer_service_calls: int
    International_plan: str
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_eve_minutes: float
    Number_vmail_messages: int
    Voice_mail_plan: str

def preprocess_data(df):
    """Preprocess the input data"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Rename columns to match training data
    column_mapping = {
        'Total_day_minutes': 'Total day minutes',
        'Customer_service_calls': 'Customer service calls',
        'International_plan': 'International plan',
        'Total_intl_minutes': 'Total intl minutes',
        'Total_intl_calls': 'Total intl calls',
        'Total_eve_minutes': 'Total eve minutes',
        'Number_vmail_messages': 'Number vmail messages',
        'Voice_mail_plan': 'Voice mail plan'
    }
    df_processed = df_processed.rename(columns=column_mapping)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_features = ['International plan', 'Voice mail plan']
    for feature in categorical_features:
        df_processed[feature] = le.fit_transform(df_processed[feature])
    
    return df_processed

@app.post("/predict")
async def predict(input_data: ChurnPredictionInput):
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Print input data for debugging
        print("Input data before processing:", df.to_dict('records'))
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Print processed data for debugging
        print("Processed data:", df_processed.to_dict('records'))
        
        # Make prediction
        prediction = model.predict(df_processed)
        probability = model.predict_proba(df_processed)[0]
        
        return {
            "churn_prediction": "Yes" if prediction[0] == 1 else "No",
            "churn_probability": float(probability[1])
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API is running. Use /predict for predictions.",
        "model_loaded": model is not None
    }
