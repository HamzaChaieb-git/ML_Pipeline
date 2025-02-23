"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any, Union, Dict
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

class TrainingCallback:
    """Custom callback to track training metrics."""
    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_accuracy": []
        }
        self.epoch = 0

    def __call__(self, env):
        """Called at each training iteration."""
        # Get training metrics
        train_loss = env.evaluation_result_list[0][2]
        
        # Store metrics
        self.history["train_loss"].append(train_loss)
        
        # Log to MLflow at each step
        mlflow.log_metric("train_loss", train_loss, step=self.epoch)
        
        # Create training curve plot every 10 epochs
        if self.epoch % 10 == 0:
            self._plot_training_curves()
        
        self.epoch += 1

    def _plot_training_curves(self):
        """Create and save training curves plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Training Loss")
        plt.title("Training Metrics Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = f"training_curve_{self.epoch}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)

def train_model(X_train: Union[pd.DataFrame, Any], y_train: Any) -> xgb.XGBClassifier:
    """Train an XGBoost model with MLflow tracking."""
    if isinstance(X_train, pd.DataFrame) and X_train.empty or y_train is None or len(y_train) == 0:
        raise ValueError("Training data or labels cannot be empty")

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": 42,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "enable_categorical": True,
        "tree_method": "hist",
        "eval_metric": ["logloss", "error"]  # Add multiple evaluation metrics
    }
    
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category')

    # Create training callback for metric tracking
    training_callback = TrainingCallback()
    eval_set = [(X_train, y_train)]

    # Train with callback
    model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=True,
        callbacks=[training_callback]
    )

    # Log final training metrics
    train_pred = model.predict(X_train)
    train_pred_proba = model.predict_proba(X_train)
    
    final_metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "train_loss": log_loss(y_train, train_pred_proba)
    }
    mlflow.log_metrics(final_metrics)

    # Log feature importance
    if isinstance(X_train, pd.DataFrame):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save and log feature importance
        feature_importance.to_csv("feature_importance.csv")
        feature_importance.to_json("feature_importance.json")
        mlflow.log_artifact("feature_importance.csv")
        mlflow.log_artifact("feature_importance.json")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.xticks(rotation=45)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")
        
        for feature, importance in zip(feature_importance['feature'], feature_importance['importance']):
            mlflow.log_metric(f"importance_{feature}", importance)

    # Log model with signature
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(
        model, 
        "model",
        signature=signature,
        input_example=X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5]
    )
    
    return model
