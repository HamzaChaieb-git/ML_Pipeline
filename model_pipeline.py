import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def prepare_data(train_file, test_file):
    """Prepare data with specific feature selection based on SHAP values."""
    with mlflow.start_run(nested=True):
        selected_features = [
            "Total day minutes", "Customer service calls",
            "International plan", "Total intl minutes",
            "Total intl calls", "Total eve minutes",
            "Number vmail messages", "Voice mail plan",
        ]
        
        # Read data
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        # Log data info
        mlflow.log_param("n_features", len(selected_features))
        mlflow.log_param("train_samples", len(df_train))
        mlflow.log_param("test_samples", len(df_test))
        
        # Prepare features and target
        X_train = df_train[selected_features]
        y_train = df_train["Churn"]
        X_test = df_test[selected_features]
        y_test = df_test["Churn"]
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_features = ["International plan", "Voice mail plan"]
        for feature in categorical_features:
            X_train[feature] = le.fit_transform(X_train[feature])
            X_test[feature] = le.transform(X_test[feature])
        
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    with mlflow.start_run(nested=True):
        model_params = {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 5,
        }
        
        # Log parameters
        mlflow.log_params(model_params)
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Log feature importances
        feature_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        feature_imp.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")
        
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics."""
    with mlflow.start_run(nested=True):
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        return metrics

def save_model(model, filename="model.joblib"):
    """Save the model using joblib and MLflow."""
    with mlflow.start_run(nested=True):
        # Save with joblib
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(filename)

def load_model(filename="model.joblib"):
    """Load the model using joblib."""
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    raise FileNotFoundError(f"Model file {filename} not found")
