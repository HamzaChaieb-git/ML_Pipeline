"""Database connector for storing predictions and monitoring data."""

import os
import pymongo
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


class MongoDBConnector:
    """Connector for MongoDB to store predictions and monitoring data."""
    
    def __init__(self, host="mongodb", port=27017, db_name="ml_monitoring"):
        """
        Initialize MongoDB connector.
        
        Args:
            host: MongoDB host address
            port: MongoDB port
            db_name: Database name to use
        """
        self.client = pymongo.MongoClient(f"mongodb://{host}:{port}/")
        self.db = self.client[db_name]
        self.predictions = self.db["predictions"]
        self.model_metrics = self.db["model_metrics"]
        self.system_metrics = self.db["system_metrics"]
        
        # Create indexes for faster queries
        self.predictions.create_index([("timestamp", pymongo.DESCENDING)])
        self.predictions.create_index([("model_version", pymongo.ASCENDING)])
        self.model_metrics.create_index([("timestamp", pymongo.DESCENDING)])
        self.model_metrics.create_index([("model_version", pymongo.ASCENDING)])
        self.system_metrics.create_index([("timestamp", pymongo.DESCENDING)])
    
    def save_prediction(self, model_version: str, features: Dict, prediction: float, 
                       actual: Optional[float] = None) -> str:
        """
        Save a prediction to MongoDB.
        
        Args:
            model_version: Model version that made the prediction
            features: Dictionary of input features
            prediction: Predicted value
            actual: Actual value (if available)
            
        Returns:
            ID of the inserted document
        """
        doc = {
            "timestamp": datetime.now(),
            "model_version": model_version,
            "features": features,
            "prediction": prediction,
            "actual": actual
        }
        
        result = self.predictions.insert_one(doc)
        return str(result.inserted_id)
    
    def save_model_metrics(self, model_version: str, metrics: Dict[str, float]) -> str:
        """
        Save model metrics to MongoDB.
        
        Args:
            model_version: Model version
            metrics: Dictionary of metrics
            
        Returns:
            ID of the inserted document
        """
        doc = {
            "timestamp": datetime.now(),
            "model_version": model_version,
            "metrics": metrics
        }
        
        result = self.model_metrics.insert_one(doc)
        return str(result.inserted_id)
    
    def save_system_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Save system metrics to MongoDB.
        
        Args:
            metrics: Dictionary of system metrics
            
        Returns:
            ID of the inserted document
        """
        doc = {
            "timestamp": datetime.now(),
            "metrics": metrics
        }
        
        result = self.system_metrics.insert_one(doc)
        return str(result.inserted_id)
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get most recent predictions."""
        return list(self.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    
    def get_model_metrics_history(self, model_version: Optional[str] = None, 
                                 limit: int = 100) -> List[Dict]:
        """Get model metrics history, optionally filtered by model version."""
        query = {}
        if model_version:
            query["model_version"] = model_version
            
        return list(self.model_metrics.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    
    def get_system_metrics_history(self, hours: int = 24, limit: int = 1000) -> List[Dict]:
        """Get system metrics for the last N hours."""
        start_time = datetime.now() - datetime.timedelta(hours=hours)
        query = {"timestamp": {"$gte": start_time}}
            
        return list(self.system_metrics.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()


# Singleton instance for application-wide use
_db_connector = None

def get_db_connector():
    """Get or create a MongoDB connector instance."""
    global _db_connector
    if _db_connector is None:
        # Try to connect to MongoDB in Docker network first
        try:
            _db_connector = MongoDBConnector(host="mongodb")
        except:
            # Fallback to localhost
            try:
                _db_connector = MongoDBConnector(host="localhost")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                return None
    
    return _db_connector