from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import mlflow
import os
import logging
from typing import List, Dict
from db_connector import get_db_connector
import json
from bson.json_util import dumps
from datetime import datetime

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
            if f.startswith("model_") and f.endswith(".joblib")
        ]
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return None

        latest_model = max(
            model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x))
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
        
        # Get model info
        model_info = getattr(model, "model_info", {"version": "unknown"})
        model_version = model_info.get("version", "unknown")
        
        # Store predictions in MongoDB
        db = get_db_connector()
        if db:
            for i, row in input_df.iterrows():
                # Convert row to dictionary for storage
                features_dict = row.to_dict()
                # Store prediction
                db.save_prediction(
                    model_version=model_version,
                    features=features_dict,
                    prediction=float(predictions[i])
                )
            logger.info("Stored predictions in MongoDB")
        else:
            logger.warning("MongoDB connection not available, predictions not stored")

        logger.info(f"Prediction successful: {predictions.tolist()}")
        return {"churn_probabilities": predictions.tolist()}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/recent-predictions")
def get_recent_predictions(limit: int = 10):
    """Get recent predictions from MongoDB."""
    db = get_db_connector()
    if not db:
        raise HTTPException(
            status_code=503, detail="Database connection not available."
        )
    
    predictions = db.get_recent_predictions(limit=limit)
    return {"recent_predictions": predictions}


@app.get("/model-metrics")
def get_model_metrics(model_version: str = None):
    """Get model performance metrics from MongoDB."""
    db = get_db_connector()
    if not db:
        raise HTTPException(
            status_code=503, detail="Database connection not available."
        )
    
    metrics = db.get_model_metrics_history(model_version=model_version, limit=100)
    return {"model_metrics": metrics}


@app.post("/debug-metrics")
def debug_metrics():
    """Debug endpoint to manually save some metrics to MongoDB for testing."""
    try:
        db = get_db_connector()
        if not db:
            return {"status": "error", "message": "MongoDB connection not available"}
        
        # Sample metrics for testing
        test_metrics = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.87,
            "f1": 0.88,
            "roc_auc": 0.95,
            "log_loss": 0.25,
            "true_positives": 150,
            "true_negatives": 800,
            "false_positives": 20, 
            "false_negatives": 30
        }
        
        # Save to MongoDB
        result = db.save_model_metrics("test_version", test_metrics)
        
        return {
            "status": "success", 
            "message": "Test metrics saved to MongoDB",
            "metrics": test_metrics,
            "document_id": result
        }
    except Exception as e:
        logger.error(f"Error in debug metrics: {str(e)}")
        return {"status": "error", "message": f"Failed to save metrics: {str(e)}"}


@app.get("/model-metrics-debug")
def get_model_metrics_debug():
    """Get model performance metrics from MongoDB with debug info."""
    try:
        db = get_db_connector()
        if not db:
            return {"status": "error", "message": "Database connection not available."}
        
        # Get all collections in the database
        collections = db.db.list_collection_names()
        
        # Count documents in the model_metrics collection
        metrics_count = db.db.model_metrics.count_documents({})
        
        # Get a sample metric document
        sample = list(db.db.model_metrics.find().limit(1))
        sample_json = json.loads(dumps(sample)) if sample else None
        
        # Get metrics history
        metrics = db.get_model_metrics_history(limit=10)
        metrics_json = json.loads(dumps(metrics)) if metrics else []
        
        return {
            "status": "success",
            "collections": collections,
            "metrics_count": metrics_count,
            "sample_document": sample_json,
            "metrics_history": metrics_json
        }
    except Exception as e:
        logger.error(f"Error in model metrics debug: {str(e)}")
        return {"status": "error", "message": f"Error retrieving metrics: {str(e)}"}


@app.post("/retrain")
def retrain_model():
    """Endpoint to trigger model retraining."""
    try:
        # This would typically call your model training script
        # For now, we'll just return a message
        return {"status": "success", "message": "Model retraining initiated"}
    except Exception as e:
        logger.error(f"Error initiating retraining: {str(e)}")
        return {"status": "error", "message": f"Failed to initiate retraining: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)