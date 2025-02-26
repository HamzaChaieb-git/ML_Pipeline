from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import mlflow
import os
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API")

# Initialize the model variable at the module level
model = None


# Load the latest model from artifacts/models/
def load_latest_model():
    try:
        models_dir = os.path.join("artifacts", "models")
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return None

        model_files = [
            f
            for f in os.listdir(models_dir)
            if f.startswith("model_v") and f.endswith(".joblib")
        ]
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return None

        latest_model = max(
            model_files, key=lambda x: x.split("v")[1].split(".joblib")[0]
        )
        model_path = os.path.join(models_dir, latest_model)

        logger.info(f"Loading model from {model_path}")
        loaded_model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {latest_model}")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None


# Try to load model on startup
try:
    model = load_latest_model()
    logger.info(f"Initial model loading status: {'Success' if model else 'Failed'}")
except Exception as e:
    logger.error(f"Error during initial model loading: {str(e)}")

# Define the expected input features (matching your training data)
expected_features = [
    "Total day minutes",
    "Customer service calls",
    "International plan",
    "Total intl minutes",
    "Total intl calls",
    "Total eve minutes",
    "Number vmail messages",
    "Voice mail plan",
]


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "status": "online",
        "service": "Churn Prediction API",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    # Simple health check that always returns healthy
    # This is separate from model status to allow the container to be considered healthy
    # even if the model hasn't been loaded yet
    return {"status": "healthy"}


@app.get("/model-status")
def model_status():
    """Model status endpoint."""
    return {
        "model_loaded": model is not None,
        "features": expected_features if model is not None else [],
    }


@app.post("/predict", response_model=Dict[str, List[float]])
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
    global model  # Moved to the beginning of the function

    # Check if model is loaded
    if model is None:
        # Try to reload the model
        model = load_latest_model()
        if model is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Service unavailable."
            )

    try:
        logger.info("Received prediction request")
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
        for col in [
            "Total day minutes",
            "Customer service calls",
            "Total intl minutes",
            "Total intl calls",
            "Total eve minutes",
            "Number vmail messages",
        ]:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        # Make predictions
        predictions = model.predict_proba(input_df)[
            :, 1
        ]  # Probability of churn (class 1)

        logger.info(f"Prediction successful: {predictions.tolist()}")
        return {"churn_probabilities": predictions.tolist()}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
