import pytest
from src.train import load_data, train_model  
from sklearn.linear_model import LinearRegression
import joblib
import os

def test_data_loading():
    """Test dataset loading"""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == 8 

def test_model_training():
    """Test model training"""
    model, r2 = train_model()
    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    assert r2 > 0.5  

def test_model_saving():
    """Test if model is saved correctly"""
    if os.path.exists("models/linear_regression.joblib"):
        os.remove("models/linear_regression.joblib")
    
    train_model()
    assert os.path.exists("models/linear_regression.joblib")
    model = joblib.load("models/linear_regression.joblib")
    assert isinstance(model, LinearRegression)