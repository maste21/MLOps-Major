# src/train.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os
from src.utils import ensure_dir

def load_data():
    """Load California housing dataset"""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    """Train and save Linear Regression model"""
    X_train, X_test, y_train, y_test = load_data()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    ensure_dir("models")
    model_path = "models/linear_regression.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, r2

if __name__ == "__main__":
    train_model()