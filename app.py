from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load the model when the app starts
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

class ChurnPredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float

@app.post("/predict")
async def predict(input_data: ChurnPredictionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0]
        
        return {
            "churn_prediction": "Yes" if prediction[0] == 1 else "No",
            "churn_probability": float(probability[1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Churn Prediction API is running. Use /predict for predictions."}
