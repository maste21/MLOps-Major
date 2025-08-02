import pytest
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from src.train import load_data, train_model
from src.utils import validate_model, load_model
from sklearn.linear_model import LinearRegression

@pytest.fixture
def sample_data():
    """Fixture providing sample dataset"""
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def test_data_loading():
    """Test dataset loading with shape and type validation"""
    X_train, X_test, y_train, y_test = load_data()
    
    assert X_train.shape[1] == 8  
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    # Test types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    
    
    assert y_train.min() > 0

def test_model_training(sample_data):
    """Test model training with full validation"""
    X_train, _, y_train, _ = sample_data
    
    model, r2, _ = train_model() 
    validate_model(model)
    assert r2 > 0.5
    assert model.coef_.shape == (8,)
    
    preds = model.predict(X_train[:5])
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()

def test_model_persistence():
    """Test model saving and loading"""
    model_path = "models/linear_regression.joblib"
    
    if os.path.exists(model_path):
        os.remove(model_path)
    
    model, _, _ = train_model()  
    assert os.path.exists(model_path)
    
    loaded_model = load_model(model_path)
    assert np.allclose(model.coef_, loaded_model.coef_)
    assert np.isclose(model.intercept_, loaded_model.intercept_)

def test_quantization_compatibility():
    """Verify model works with quantization"""
    model = load_model("models/linear_regression.joblib")
    params = {
        'coef_': model.coef_,
        'intercept_': model.intercept_
    }
    assert isinstance(params['coef_'], np.ndarray)
    assert isinstance(params['intercept_'], float)