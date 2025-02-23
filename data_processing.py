"""Module for processing and preparing machine learning data."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def prepare_data(train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and testing data with specific feature selection based on SHAP values.

    Args:
        train_file (str): Path to the training CSV file.
        test_file (str): Path to the testing CSV file.

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test) as DataFrames and Series.

    Raises:
        FileNotFoundError: If the input CSV files are not found.
    """
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

    X_train = df_train[selected_features].copy()  # Avoid view/copy issues
    y_train = df_train["Churn"]

    X_test = df_test[selected_features].copy()
    y_test = df_test["Churn"]

    le = LabelEncoder()

    categorical_features = ["International plan", "Voice mail plan"]
    for feature in categorical_features:
        X_train.loc[:, feature] = le.fit_transform(X_train[feature])
        X_test.loc[:, feature] = le.transform(X_test[feature])

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test
