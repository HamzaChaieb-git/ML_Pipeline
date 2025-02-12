import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data(train_path, test_path, target_column='Churn'):
    """Load and preprocess the dataset.
    
    Args:
        train_path (str): Path to the training dataset CSV.
        test_path (str): Path to the test dataset CSV.
        target_column (str): Name of the target variable.

    Returns:
        X_train, X_test, y_train, y_test: Processed feature and target sets.
    """
    
    # Load dataset
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Identify categorical columns
    categorical_cols = df_train.select_dtypes(include=['object']).columns

    # Convert categorical columns to numerical
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_train[col] = label_encoders[col].fit_transform(df_train[col])
        df_test[col] = label_encoders[col].transform(df_test[col])

    # Define features and target variable
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    return X_train, X_test, y_train, y_test


from xgboost import XGBClassifier

def train_model(X_train, y_train):
    """Train an XGBoost classifier model.
    
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target variable.

    Returns:
        model: Trained XGBoost model.
    """
    
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    return model



from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance.
    
    Args:
        model: Trained model.
        X_test (DataFrame): Test features.
        y_test (Series): True test labels.

    Returns:
        float: Accuracy score of the model.
    """
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    return accuracy
import pickle

def save_model(model, filename="model.pkl"):
    """Save the trained model to a file.
    
    Args:
        model: Trained model.
        filename (str): File name to save the model.
    """
    
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f'Model saved as {filename}')

import pickle

def load_model(filename="model.pkl"):
    """Load a saved model from a file.
    
    Args:
        filename (str): File name of the saved model.

    Returns:
        model: Loaded model.
    """
    
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    
    return model

