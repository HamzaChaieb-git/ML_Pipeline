"""Module for training machine learning models using XGBoost and MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any, Union, Dict
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from xgboost.callback import TrainingCallback

class MetricTracker(TrainingCallback):
    """Custom callback to track training metrics."""
    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.eval_loss_history = []
        self.iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration."""
        if evals_log:
            # Get the latest evaluation metrics
            train_loss = evals_log.get('train', {}).get('logloss', [])[-1] if 'train' in evals_log else None
            eval_loss = evals_log.get('eval', {}).get('logloss', [])[-1] if 'eval' in evals_log else None

            # Store metrics
            if train_loss is not None:
                self.train_loss_history.append(train_loss)
                mlflow.log_metric("train_loss", train_loss, step=self.iteration)

            if eval_loss is not None:
                self.eval_loss_history.append(eval_loss)
                mlflow.log_metric("eval_loss", eval_loss, step=self.iteration)

            # Plot training curve every 10 iterations
            if self.iteration % 10 == 0:
                self._plot_training_curve()

            self.iteration += 1
        return False

    def _plot_training_curve(self):
        """Create and log training curve plot."""
        plt.figure(figsize=(10, 6))
        
        if self.train_loss_history:
            plt.plot(self.train_loss_history, label='Training Loss', color='blue')
        if self.eval_loss_history:
            plt.plot(self.eval_loss_history, label='Validation Loss', color='red')
        
        plt.title('Training Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = f"training_curve_{self.iteration}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)

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
        "eval_metric": ["logloss", "error"]
    }
    
    mlflow.log_params(params)
    mlflow.log_param("model_version", model_version)

    model = xgb.XGBClassifier(**params)
    
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train_final[col] = X_train_final[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    # Initialize callback
    metric_tracker = MetricTracker()
    
    # Train with validation set and callback
    model.fit(
        X_train_final, 
        y_train_final,
        eval_set=[(X_train_final, y_train_final), (X_val, y_val)],
        callbacks=[metric_tracker],
        verbose=True
    )

    # Create and log final feature importance plot
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
