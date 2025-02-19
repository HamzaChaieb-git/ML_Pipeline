import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def prepare_data(train_file, test_file):
    """
    Prepare data with specific feature selection based on SHAP values
    """
    # Define selected features
    selected_features = [
        'Total day minutes',
        'Customer service calls',
        'International plan',
        'Total intl minutes',
        'Total intl calls',
        'Total eve minutes',
        'Number vmail messages',
        'Voice mail plan'
    ]
    
    # Read the training and testing data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Separate features and target
    X_train = df_train[selected_features]
    y_train = df_train['Churn']
    
    X_test = df_test[selected_features]
    y_test = df_test['Churn']
    
    # Encode categorical variables
    le = LabelEncoder()
    
    # Handle categorical features
    categorical_features = ['International plan', 'Voice mail plan']
    for feature in categorical_features:
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
    
    # Encode target variable
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    from sklearn.metrics import classification_report, confusion_matrix
    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def save_model(model, filename='model.joblib'):
    """Save the model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename='model.joblib'):
    """Load the model"""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        raise FileNotFoundError(f"Model file {filename} not found")
