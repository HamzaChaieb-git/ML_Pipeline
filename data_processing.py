"""Module for processing and preparing machine learning data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import mlflow


def prepare_data(train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare training and testing data with specific feature selection based on SHAP values.

    Args:
        train_file (str): Path to the training CSV file.
        test_file (str): Path to the testing CSV file.

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test) as DataFrames and NumPy arrays.

    Raises:
        FileNotFoundError: If the input CSV files are not found.
        ValueError: If required columns are missing in the data.
    """
    print("🔹 Preparing data...")
    selected_features = [
        "Total day minutes",
        "Customer service calls",
        "International plan",
        "Total intl minutes",
        "Total intl calls",
        "Total eve minutes",
        "Number vmail messages",
        "Voice mail plan",
    ]

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data files: {e}")

    # Log data info
    mlflow.log_param("train_samples", len(df_train))
    mlflow.log_param("test_samples", len(df_test))
    
    # Check for missing columns
    missing_cols = [col for col in selected_features + ["Churn"] if col not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in training data: {missing_cols}")

    X_train = df_train[selected_features].copy()
    y_train = df_train["Churn"].copy()

    X_test = df_test[selected_features].copy()
    y_test = df_test["Churn"].copy()

    # Handle categorical features
    categorical_features = ["International plan", "Voice mail plan"]
    label_encoders = {}
    
    for feature in categorical_features:
        # Create and fit label encoder
        le = LabelEncoder()
        # Combine train and test data to fit encoder
        all_values = pd.concat([X_train[feature], X_test[feature]])
        le.fit(all_values)
        
        # Transform both train and test
        X_train[feature] = le.transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
        
        # Store the encoder
        label_encoders[feature] = le
        
        # Log the encoding mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mlflow.log_dict(mapping, f"encoding_map_{feature}.json")

    # Handle numeric features
    numeric_features = [
        "Total day minutes",
        "Customer service calls",
        "Total intl minutes",
        "Total intl calls",
        "Total eve minutes",
        "Number vmail messages"
    ]
    
    for feature in numeric_features:
        X_train[feature] = pd.to_numeric(X_train[feature], errors='coerce')
        X_test[feature] = pd.to_numeric(X_test[feature], errors='coerce')
        
        # Log feature statistics
        stats = {
            "mean": float(X_train[feature].mean()),
            "std": float(X_train[feature].std()),
            "min": float(X_train[feature].min()),
            "max": float(X_train[feature].max())
        }
        mlflow.log_dict(stats, f"feature_stats_{feature}.json")

    # Convert target to numeric
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.transform(y_test)
    
    # Log target encoding mapping
    target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
    mlflow.log_dict(target_mapping, "target_encoding_map.json")

    # Log feature names and types
    feature_info = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "target": "Churn"
    }
    mlflow.log_dict(feature_info, "feature_info.json")

    print("🔹 Data preparation complete")
    return X_train, X_test, y_train, y_test
