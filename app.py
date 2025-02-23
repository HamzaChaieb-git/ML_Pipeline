from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import subprocess
import tempfile

app = FastAPI()

# Initialize model as None
model = None

def pull_model_from_docker():
    """Pull the latest model from Docker container"""
    try:
        # Create a temporary container from the image
        container_id = subprocess.check_output(
            ["docker", "create", "hamzachaieb01/ml-trained:latest"],
            text=True
        ).strip()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            
            # Copy the model file from the container
            subprocess.run([
                "docker", "cp",
                f"{container_id}:/app/model.pkl",
                model_path
            ], check=True)
            
            # Load the model
            with open(model_path, 'rb') as f:
                model_obj = pickle.load(f)
            
            # Clean up the container
            subprocess.run(["docker", "rm", container_id], check=True)
            
            return model_obj
            
    except Exception as e:
        print(f"Error pulling model from Docker: {e}")
        raise Exception(f"Failed to pull model from Docker: {e}")

# Load the model when the app starts
def load_model():
    global model
    try:
        model = pull_model_from_docker()
        print("Model loaded successfully from Docker")
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
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
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

# Rest of the code remains the same...
