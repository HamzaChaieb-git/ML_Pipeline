import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.xgboost

def prepare_data(train_file, test_file):
    """
    Prepare data with specific feature selection based on SHAP values.
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

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    X_train = df_train[selected_features]
    y_train = df_train["Churn"]

    X_test = df_test[selected_features]
    y_test = df_test["Churn"]

    le = LabelEncoder()

    categorical_features = ["International plan", "Voice mail plan"]
    for feature in categorical_features:
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the XGBoost model and log with MLflow.
    """
    with mlflow.start_run():
        params = {
            "objective": "binary:logistic",  # Binary classification
            "max_depth": 6,                 # Maximum depth of trees
            "learning_rate": 0.1,           # Step size shrinkage
            "n_estimators": 100,            # Number of boosting rounds
            "random_state": 42,             # For reproducibility
            "min_child_weight": 1,          # Minimum sum of instance weight needed in a child
            "subsample": 0.8,              # Subsample ratio of the training instance
            "colsample_bytree": 0.8,       # Subsample ratio of columns when constructing each tree
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        mlflow.xgboost.log_model(model, "xgboost_model")
        
        return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model, log metrics, and print classification metrics.
    """
    with mlflow.start_run():
        predictions = model.predict(X_test)
        
        report = classification_report(y_test, predictions, output_dict=True)
        confusion = confusion_matrix(y_test, predictions)
        
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "precision_0": report['0']['precision'],
            "recall_0": report['0']['recall'],
            "f1_0": report['0']['f1-score'],
            "precision_1": report['1']['precision'],
            "recall_1": report['1']['recall'],
            "f1_1": report['1']['f1-score'],
        })

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion)

def save_model(model, filename="model.joblib"):
    """
    Save the model using joblib and log it with MLflow.
    """
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
    with mlflow.start_run():
        mlflow.log_artifact(filename)

def load_model(filename="model.joblib"):
    """
    Load the model using joblib.
    """
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    raise FileNotFoundError(f"Model file {filename} not found")
