from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import mlflow
import os
from typing import List, Dict

app = FastAPI(title="Churn Prediction API")

# Load the model from MLflow or local storage
def load_model(model_path: str = "artifacts/models/model_v20250224_1032.joblib"):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Global model instance (loaded once on startup)
model = load_model()

# Define the expected input features (matching your training data)
expected_features = [
    "Total day minutes",
    "Customer service calls",
    "International plan",
    "Total intl minutes",
    "Total intl calls",
    "Total eve minutes",
    "Number vmail messages",
    "Voice mail plan"
]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=Dict[str, float])
def predict(churn_data: Dict[str, List[float]]):
    """
    Predict churn probability for a list of customer features.

    Example input:
    {
        "Total day minutes": [120.5, 150.3, ...],
        "Customer service calls": [3, 2, ...],
        "International plan": [0, 1, ...],  # 0 for No, 1 for Yes
        "Total intl minutes": [10.2, 8.5, ...],
        "Total intl calls": [5, 4, ...],
        "Total eve minutes": [200.0, 180.5, ...],
        "Number vmail messages": [0, 5, ...],
        "Voice mail plan": [0, 1, ...]  # 0 for No, 1 for Yes
    }
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame(churn_data)
        
        # Validate input features
        if not all(feature in input_df.columns for feature in expected_features):
            missing = [f for f in expected_features if f not in input_df.columns]
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        # Ensure the order and types match the training data
        input_df = input_df[expected_features]
        for col in ["International plan", "Voice mail plan"]:
            input_df[col] = input_df[col].astype(int)
        for col in ["Total day minutes", "Customer service calls", "Total intl minutes", 
                   "Total intl calls", "Total eve minutes", "Number vmail messages"]:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Make predictions
        predictions = model.predict_proba(input_df)[:, 1]  # Probability of churn (class 1)
        
        # Return predictions as a dictionary
        return {"churn_probabilities": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
