from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Form, File, UploadFile
import pandas as pd
import joblib
import mlflow
import os
import logging
from typing import List, Dict, Any, Optional
from db_connector import get_db_connector
import json
from bson.json_util import dumps
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API")

# Initialize the model variable at the module level
model = None

# Create Basic Prometheus metrics
PREDICTIONS_COUNTER = Counter('ml_predictions_total', 'Total number of predictions', ['model_version', 'prediction_class'])
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Time spent processing prediction', ['model_version'])
PREDICTION_SCORES = Histogram('ml_prediction_scores', 'Distribution of prediction scores', ['model_version'])
MODEL_LOADED = Gauge('ml_model_loaded', 'Model loading status (1=loaded, 0=not loaded)')

# Advanced metrics
PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence', 
    'Distribution of model confidence scores', 
    ['model_version', 'prediction_class'],
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0]
)

FEATURE_VALUES = Histogram(
    'ml_feature_values', 
    'Distribution of input feature values', 
    ['model_version', 'feature_name'],
    buckets=[float('-inf'), 0, 10, 50, 100, 200, 500, float('inf')]
)

MODEL_RELOADS = Counter(
    'ml_model_reloads_total', 
    'Number of times the model was reloaded', 
    ['model_version', 'reason']
)

PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total', 
    'Number of errors during prediction', 
    ['model_version', 'error_type']
)

PREDICTION_BATCH_SIZE = Histogram(
    'ml_prediction_batch_size', 
    'Size of prediction batches', 
    ['model_version'],
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500]
)

REQUEST_SIZE = Histogram(
    'ml_request_payload_bytes', 
    'Size of prediction request payloads', 
    ['model_version'],
    buckets=[10, 100, 1000, 10000, 100000, 1000000]
)

# Load the latest model from artifacts/models/
def load_latest_model():
    try:
        models_dir = os.path.join("artifacts", "models")
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            MODEL_LOADED.set(0.0)
            return None

        model_files = [
            f
            for f in os.listdir(models_dir)
            if f.startswith("model_") and f.endswith(".joblib")
        ]
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            MODEL_LOADED.set(0.0)
            return None

        latest_model = max(
            model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x))
        )
        model_path = os.path.join(models_dir, latest_model)

        logger.info(f"Loading model from {model_path}")
        loaded_model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {latest_model}")
        MODEL_LOADED.set(1.0)
        
        # Extract model version from filename
        model_version = latest_model.replace("model_", "").replace(".joblib", "")
        MODEL_RELOADS.labels(model_version=model_version, reason="startup").inc()
        
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL_LOADED.set(0.0)
        return None


# Try to load model on startup
try:
    model = load_latest_model()
    logger.info(f"Initial model loading status: {'Success' if model else 'Failed'}")
    MODEL_LOADED.set(1.0 if model else 0.0)
except Exception as e:
    logger.error(f"Error during initial model loading: {str(e)}")
    MODEL_LOADED.set(0.0)

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


@app.get('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


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
    start_time = time.time()

    # Check if model is loaded
    if model is None:
        # Try to reload the model
        model = load_latest_model()
        if model is None:
            PREDICTION_ERRORS.labels(model_version="unknown", error_type="model_not_loaded").inc()
            raise HTTPException(
                status_code=503, detail="Model not loaded. Service unavailable."
            )

    try:
        logger.info("Received prediction request")
        # Get model info
        model_info = getattr(model, "model_info", {"version": "unknown"})
        model_version = model_info.get("version", "unknown")
        
        # Track request payload size
        request_size = len(str(churn_data))
        REQUEST_SIZE.labels(model_version=model_version).observe(request_size)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame(churn_data)
        
        # Track batch size
        batch_size = len(input_df)
        PREDICTION_BATCH_SIZE.labels(model_version=model_version).observe(batch_size)

        # Validate input features
        if not all(feature in input_df.columns for feature in expected_features):
            missing = [f for f in expected_features if f not in input_df.columns]
            PREDICTION_ERRORS.labels(model_version=model_version, error_type="missing_features").inc()
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
            
            # Track feature value distributions
            for value in input_df[col]:
                FEATURE_VALUES.labels(model_version=model_version, feature_name=col).observe(value)

        # Make predictions
        predictions = model.predict_proba(input_df)[:, 1]  # Probability of churn (class 1)
        
        # Record metrics
        # Record prediction latency
        PREDICTION_LATENCY.labels(model_version=model_version).observe(time.time() - start_time)
        
        # Record prediction counts by class
        for pred in predictions:
            class_label = "churn" if pred > 0.5 else "no_churn"
            PREDICTIONS_COUNTER.labels(model_version=model_version, prediction_class=class_label).inc()
            
            # Track prediction confidence
            confidence = pred if class_label == "churn" else 1 - pred
            PREDICTION_CONFIDENCE.labels(model_version=model_version, prediction_class=class_label).observe(confidence)
        
        # Record prediction score distribution
        for pred in predictions:
            PREDICTION_SCORES.labels(model_version=model_version).observe(pred)
        
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
        model_version = getattr(model, "model_info", {}).get("version", "unknown") if model else "unknown"
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="prediction_failure").inc()
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
async def retrain_model_endpoint(
    train_file: str = Form(...),
    test_file: str = Form(...),
    auto_promote: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """
    Endpoint to trigger model retraining using specified datasets.
    
    Args:
        train_file: Path to the training CSV file
        test_file: Path to the testing CSV file
        auto_promote: Whether to automatically promote the model if metrics are good
        
    Returns:
        Dictionary with retraining status and information
    """
    from model_retrain import retrain_model
    
    logger.info(f"Retraining requested with: train_file={train_file}, test_file={test_file}, auto_promote={auto_promote}")
    
    # Validate file paths
    if not os.path.exists(train_file):
        raise HTTPException(status_code=400, detail=f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise HTTPException(status_code=400, detail=f"Test file not found: {test_file}")
    
    # Function for background retraining
    def _retrain_in_background():
        try:
            success, result = retrain_model(train_file, test_file, auto_promote)
            # Store result in a file for later retrieval
            os.makedirs("artifacts/retraining", exist_ok=True)
            result_file = os.path.join("artifacts", "retraining", f"retrain_{result.get('model_version', 'unknown')}.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Retraining completed with status: {result.get('status')}")
            
            # Update MongoDB if available
            try:
                db = get_db_connector()
                if db:
                    # Log metrics if available
                    if "metrics" in result and result["metrics"]:
                        db.save_model_metrics(result.get("model_version", "unknown"), result["metrics"])
                        logger.info(f"Logged model metrics to MongoDB")
            except Exception as e:
                logger.error(f"Failed to update MongoDB after retraining: {e}")
        except Exception as e:
            logger.error(f"Background retraining task failed: {e}")
    
    # Start retraining in the background if background_tasks is provided
    if background_tasks:
        background_tasks.add_task(_retrain_in_background)
        return {
            "status": "started",
            "message": "Retraining started in the background. Use /retrain-status to check progress.",
            "train_file": train_file,
            "test_file": test_file,
            "auto_promote": auto_promote,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Run synchronously
        success, result = retrain_model(train_file, test_file, auto_promote)
        return result


@app.get("/retrain-status")
async def retrain_status(version: Optional[str] = None):
    """
    Get the status of the most recent retraining job or a specific version.
    
    Args:
        version: Optional specific model version to check
        
    Returns:
        Dictionary with retraining status and information
    """
    try:
        retraining_dir = os.path.join("artifacts", "retraining")
        if not os.path.exists(retraining_dir):
            return {"status": "no_data", "message": "No retraining data available"}
        
        # Get all retraining result files
        result_files = [f for f in os.listdir(retraining_dir) if f.startswith("retrain_") and f.endswith(".json")]
        if not result_files:
            return {"status": "no_data", "message": "No retraining data available"}
        
        if version:
            # Try to find the specific version requested
            matching_files = [f for f in result_files if version in f]
            if not matching_files:
                return {"status": "not_found", "message": f"No retraining data found for version {version}"}
            
            result_file = os.path.join(retraining_dir, matching_files[0])
        else:
            # Get the most recent result file based on modification time
            result_file = os.path.join(retraining_dir, max(
                result_files, key=lambda x: os.path.getmtime(os.path.join(retraining_dir, x))
            ))
        
        # Load the retraining result
        with open(result_file, "r") as f:
            result = json.load(f)
        
        return result
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/available-datasets")
async def available_datasets():
    """
    Get information about available datasets for retraining.
    
    Returns:
        Dictionary with dataset information
    """
    from model_retrain import get_available_datasets
    return get_available_datasets()


@app.get("/registered-models")
async def registered_models():
    """
    Get information about registered models.
    
    Returns:
        Dictionary with model information
    """
    from model_retrain import get_registered_models
    return get_registered_models()


@app.post("/promote-model")
async def promote_model(model_name: str = Form(...), version: str = Form(...)):
    """
    Endpoint to manually promote a model to production.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote
        
    Returns:
        Dictionary with promotion result
    """
    from model_retrain import manual_promote_model
    return manual_promote_model(model_name, version)


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form(...)  # Either "train" or "test"
):
    """
    Upload a dataset for model retraining.
    
    Args:
        file: CSV file to upload
        dataset_type: Type of dataset (train or test)
        
    Returns:
        Dictionary with upload result
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs("artifacts/data", exist_ok=True)
        
        # Determine filename
        if dataset_type.lower() not in ["train", "test"]:
            raise HTTPException(status_code=400, detail="dataset_type must be 'train' or 'test'")
        
        # Get original filename and extension
        filename_parts = os.path.splitext(file.filename)
        if filename_parts[1].lower() != ".csv":
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        target_filename = f"{dataset_type}_{timestamp}.csv"
        file_path = os.path.join("artifacts", "data", target_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Try to verify it's a valid CSV with the right columns
        try:
            df = pd.read_csv(file_path, nrows=5)
            has_target = "Churn" in df.columns
            if not has_target:
                os.remove(file_path)  # Delete the invalid file
                raise HTTPException(status_code=400, detail="CSV file must contain a 'Churn' column")
            
            # Get data stats
            row_count = len(pd.read_csv(file_path))
            column_count = len(df.columns)
        except Exception as e:
            os.remove(file_path)  # Delete the invalid file
            raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Dataset uploaded successfully as {target_filename}",
            "filename": file_path,
            "dataset_type": dataset_type,
            "rows": row_count,
            "columns": column_count,
            "has_target": has_target
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)