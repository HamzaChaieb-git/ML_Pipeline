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
    }
    
    mlflow.log_params(params)
    mlflow.log_param("model_version", model_version)

    model = xgb.XGBClassifier(**params)
    
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train_final[col] = X_train_final[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    # Train with validation set and collect eval results
    eval_results = {}
    model.fit(
        X_train_final, 
        y_train_final,
        eval_set=[(X_train_final, y_train_final), (X_val, y_val)],
        verbose=True,
        eval_metric=['logloss', 'error'],
        evals_result=eval_results
    )

    # Plot and log training progress
    plt.figure(figsize=(12, 5))
    
    # Plot training metrics
    epochs = len(eval_results['validation_0']['logloss'])
    x_axis = range(epochs)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, eval_results['validation_0']['logloss'], 'b-', label='Train')
    plt.plot(x_axis, eval_results['validation_1']['logloss'], 'r-', label='Validation')
    plt.title('XGBoost Log Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, eval_results['validation_0']['error'], 'b-', label='Train')
    plt.plot(x_axis, eval_results['validation_1']['error'], 'r-', label='Validation')
    plt.title('XGBoost Classification Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Log training curves
    mlflow.log_artifact('training_curves.png')

    # Log each metric separately in MLflow
    for i, (train_loss, val_loss) in enumerate(zip(
        eval_results['validation_0']['logloss'],
        eval_results['validation_1']['logloss']
    )):
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=i)

    # Create and log feature importance plot
    if isinstance(X_train, pd.DataFrame):
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save and log
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")
        
        # Log feature importance values
        importance_df.to_csv("feature_importance.csv")
        mlflow.log_artifact("feature_importance.csv")
        
        for feature, importance in zip(importance_df['feature'], importance_df['importance']):
            mlflow.log_metric(f"importance_{feature}", importance)

    # Log final model
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(
        model, 
        "model",
        signature=signature,
        input_example=X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5],
        registered_model_name=f"churn_prediction_model_v{model_version}"
    )
    
    return model
