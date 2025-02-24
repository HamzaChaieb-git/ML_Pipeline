"""Enhanced module for training machine learning models with comprehensive MLflow tracking."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Any, Union, Dict, Tuple
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

class TrainingArtifacts:
    def __init__(self, model_version: str):
        """Initialize training artifacts manager."""
        self.version = model_version
        self.artifact_dir = os.path.join("artifacts", "training", f"{model_version}")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def log_data_stats(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Log comprehensive data statistics."""
        stats = {
            "dataset_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "feature_names": X.columns.tolist(),
                "class_distribution": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
                "missing_values": {str(k): int(v) for k, v in X.isnull().sum().to_dict().items()}
            },
            "feature_stats": {
                col: {
                    "mean": float(X[col].mean()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    "std": float(X[col].std()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    "min": float(X[col].min()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    "max": float(X[col].max()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    "unique_values": int(X[col].nunique()) if pd.api.types.is_numeric_dtype(X[col]) else None
                } for col in X.columns
            }
        }
        
        with open(os.path.join(self.artifact_dir, "data_statistics.json"), "w") as f:
            json.dump(stats, f, indent=4)
        mlflow.log_artifact(os.path.join(self.artifact_dir, "data_statistics.json"))

    def create_correlation_matrix(self, X: pd.DataFrame) -> None:
        """Create and log correlation matrix visualizations."""
        # Calculate correlations for numeric columns only
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            corr_matrix = X[numeric_cols].corr()
            
            # Static correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.artifact_dir, "correlation_matrix.png"))
            plt.close()
            
            # Interactive correlation heatmap
            fig = px.imshow(corr_matrix,
                           labels=dict(color="Correlation"),
                           title="Interactive Feature Correlation Matrix")
            fig.write_html(os.path.join(self.artifact_dir, "correlation_matrix.html"))
            
            mlflow.log_artifact(os.path.join(self.artifact_dir, "correlation_matrix.png"))
            mlflow.log_artifact(os.path.join(self.artifact_dir, "correlation_matrix.html"))

    def plot_feature_distributions(self, X: pd.DataFrame) -> None:
        """Create and log feature distribution plots."""
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create distribution plots for numeric features
        for col in numeric_cols:
            fig = px.histogram(X, x=col, title=f'{col} Distribution')
            fig.write_html(os.path.join(self.artifact_dir, f"dist_{col}.html"))
        
        mlflow.log_artifacts(self.artifact_dir)

    def log_training_progress(self, results: Dict) -> None:
        """Log interactive training progress visualization."""
        if not results:
            return
            
        fig = go.Figure()
        
        for metric_name, values in results.items():
            if isinstance(values, dict):
                for eval_name, eval_values in values.items():
                    fig.add_trace(go.Scatter(
                        y=eval_values,
                        name=f"{metric_name}-{eval_name}",
                        mode='lines'
                    ))
        
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Iteration',
            yaxis_title='Metric Value',
            height=600
        )
        fig.write_html(os.path.join(self.artifact_dir, "training_progress.html"))
        mlflow.log_artifact(os.path.join(self.artifact_dir, "training_progress.html"))

def train_model(X_train: Union[pd.DataFrame, Any], y_train: Any, model_version: str = "1.0.0") -> xgb.XGBClassifier:
    """
    Train an XGBoost model with comprehensive MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_version: Model version string
    
    Returns:
        Trained XGBoost classifier
    """
    # Input validation
    if isinstance(X_train, pd.DataFrame) and X_train.empty or y_train is None or len(y_train) == 0:
        raise ValueError("Training data or labels cannot be empty")

    # Initialize artifacts manager
    artifacts = TrainingArtifacts(model_version)
    
    # Log data statistics and visualizations
    if isinstance(X_train, pd.DataFrame):
        artifacts.log_data_stats(X_train, y_train)
        artifacts.create_correlation_matrix(X_train)
        artifacts.plot_feature_distributions(X_train)

    # Create validation split
    X_val = X_train.sample(frac=0.2, random_state=42)
    y_val = y_train[X_val.index]
    X_train_final = X_train.drop(X_val.index)
    y_train_final = y_train[X_train_final.index]

    # Define hyperparameters with explanations
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,             # Maximum depth of trees
        "learning_rate": 0.1,       # Learning rate / eta
        "n_estimators": 100,        # Number of boosting rounds
        "min_child_weight": 1,      # Minimum sum of instance weight in a child
        "subsample": 0.8,           # Subsample ratio of training instances
        "colsample_bytree": 0.8,    # Subsample ratio of columns for each tree
        "gamma": 0.1,               # Minimum loss reduction for partition
        "reg_alpha": 0.1,           # L1 regularization
        "reg_lambda": 1,            # L2 regularization
        "scale_pos_weight": 1,      # Class weight balance
        "enable_categorical": True,  # Enable categorical feature support
        "tree_method": "hist",      # Use histogram-based algorithm
        "eval_metric": ["error", "logloss", "auc"],  # Evaluation metrics
        "use_label_encoder": False  # Disable label encoding warning
    }
    
    # Log parameters and their descriptions
    param_desc = {
        "max_depth": "Controls the maximum depth of each tree, higher value means more complex model",
        "learning_rate": "Step size shrinkage used to prevent overfitting",
        "n_estimators": "Number of gradient boosted trees, higher value means more powerful model",
        "min_child_weight": "Minimum sum of instance weight needed in a child to continue partitioning",
        "subsample": "Percentage of samples used for training each tree",
        "colsample_bytree": "Percentage of features used for training each tree",
        "gamma": "Minimum loss reduction required to make a further partition",
        "reg_alpha": "L1 regularization term on weights",
        "reg_lambda": "L2 regularization term on weights",
        "scale_pos_weight": "Controls the balance of positive and negative weights"
    }
    
    mlflow.log_params(params)
    with open(os.path.join(artifacts.artifact_dir, "parameter_descriptions.json"), "w") as f:
        json.dump(param_desc, f, indent=4)
    mlflow.log_artifact(os.path.join(artifacts.artifact_dir, "parameter_descriptions.json"))

    # Initialize model
    model = xgb.XGBClassifier(**params)
    
    # Handle categorical features
    if isinstance(X_train, pd.DataFrame):
        categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            X_train_final[col] = X_train_final[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    # Create evaluation set
    eval_set = [(X_train_final, y_train_final), (X_val, y_val)]

    # Train model
    model.fit(
        X_train_final,
        y_train_final,
        eval_set=eval_set,
        verbose=True
    )

    # Log training progress
    if hasattr(model, 'evals_result'):
        artifacts.log_training_progress(model.evals_result())

    # Create and log feature importance plots
    if isinstance(X_train, pd.DataFrame):
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        # Static feature importance plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts.artifact_dir, "feature_importance.png"))
        plt.close()

        # Interactive feature importance plot
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title='Feature Importance (Interactive)')
        fig.write_html(os.path.join(artifacts.artifact_dir, "feature_importance.html"))
        
        mlflow.log_artifact(os.path.join(artifacts.artifact_dir, "feature_importance.png"))
        mlflow.log_artifact(os.path.join(artifacts.artifact_dir, "feature_importance.html"))
        for feature, importance in zip(importance_df['feature'], importance_df['importance']):
            mlflow.log_metric(f"feature_importance_{feature}", importance)

    # Calculate and log training metrics
    train_pred = model.predict(X_train_final)
    train_pred_proba = model.predict_proba(X_train_final)
    
    metrics = {
        "train_accuracy": accuracy_score(y_train_final, train_pred),
        "train_logloss": log_loss(y_train_final, train_pred_proba)
    }
    mlflow.log_metrics(metrics)

    # Log the model with signature
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5],
        registered_model_name=f"churn_prediction_model_v{model_version}"
    )

    return model
