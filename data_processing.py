"""Module for processing and preparing machine learning data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


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

    # Check for missing columns
    missing_cols = [col for col in selected_features + ["Churn"] if col not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in training data: {missing_cols}")

    X_train = df_train[selected_features].copy()
    y_train = df_train["Churn"].copy()

    X_test = df_test[selected_features].copy()
    y_test = df_test["Churn"].copy()

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

    # Encode categorical features
    categorical_features = ["International plan", "Voice mail plan"]
    for feature in categorical_features:
        X_train[feature] = X_train[feature].astype('category')
        X_test[feature] = X_test[feature].astype('category')

    # Convert target to numeric
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("🔹 Data preparation complete")
    return X_train, X_test, y_train, y_test
