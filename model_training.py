"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any, Union, Dict
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

def train_model(X_train: Union[pd.DataFrame, Any], y_train: Any, model_version: str = "1.0.0") -> xgb.XGBClassifier:
    """Train an XGBoost model with MLflow tracking."""
    if isinstance(X_train, pd.DataFrame) and X_train.empty or y_train is None or len(y_train) == 0:
        raise ValueError("Training data or labels cannot be empty")

    # Create evaluation dataset
    X_val = X_train.sample(frac=0.2, random_state=42)
    y_val = y_train[X_val.index]
    X_train_final = X_train.drop(X_val.index)
    y_train_final = y_train[X_train_final.index]

    # Define model parameters
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
        "use_label_encoder": False
    }
    
    # Log parameters to MLflow
    mlflow.log_params(params)
    mlflow.log_param("model_version", model_version)

    # Initialize model
    model = xgb.XGBClassifier(**params)
    
    # Handle categorical features
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train_final[col] = X_train_final[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    # Create eval set
    eval_set = [(X_train_final, y_train_final), (X_val, y_val)]
    eval_labels = ['train', 'validation']

    # Train the model
    model.fit(
        X_train_final, 
        y_train_final,
        eval_set=eval_set,
        verbose=True
    )

    # Get the evaluation results
    results = model.evals_result()

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot training metrics if available
    if results:
        for i, metric in enumerate(['validation_0-error', 'validation_1-error']):
            if metric in results:
                plt.plot(results[metric], 
                        label=f"{eval_labels[i]} error",
                        alpha=0.8)
                
                # Log metrics to MLflow
                for step, value in enumerate(results[metric]):
                    mlflow.log_metric(f"{eval_labels[i]}_error", value, step=step)

    plt.title('Training Metrics')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    # Save and log the plot
    plt.savefig('training_curves.png')
    plt.close()
    mlflow.log_artifact('training_curves.png')

    # Create feature importance plot
    if isinstance(X_train, pd.DataFrame):
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save and log feature importance
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")
        
        # Log feature importance as CSV
        importance_df.to_csv("feature_importance.csv")
        mlflow.log_artifact("feature_importance.csv")
        
        # Log individual feature importance scores
        for feature, importance in zip(importance_df['feature'], importance_df['importance']):
            mlflow.log_metric(f"importance_{feature}", importance)

    # Calculate and log training metrics
    train_pred = model.predict(X_train_final)
    train_pred_proba = model.predict_proba(X_train_final)
    
    metrics = {
        "train_accuracy": accuracy_score(y_train_final, train_pred),
        "train_logloss": log_loss(y_train_final, train_pred_proba)
    }
    mlflow.log_metrics(metrics)

    # Log the model
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(
        model, 
        "model",
        signature=signature,
        input_example=X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5],
        registered_model_name=f"churn_prediction_model_v{model_version}"
    )
    
    return model
