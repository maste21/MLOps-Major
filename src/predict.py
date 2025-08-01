# src/predict.py
import joblib
from sklearn.datasets import fetch_california_housing
import numpy as np

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load("models/linear_regression.joblib")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(model, X):
    """Make predictions using the model"""
    if model is None:
        return None
    return model.predict(X)

if __name__ == "__main__":
    # Load model
    model = load_model()
    if model is None:
        exit(1)
    
    # Load sample data
    data = fetch_california_housing()
    X_sample = data.data[:5]  # First 5 samples
    
    # Make predictions
    predictions = make_predictions(model, X_sample)
    
    print("Sample predictions:")
    for i, (features, pred) in enumerate(zip(X_sample, predictions)):
        print(f"Sample {i+1}:")
        print(f"  Features: {features}")
        print(f"  Prediction: {pred:.2f}")
        print()