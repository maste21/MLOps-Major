import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

def ensure_dir(directory):
    """Create directory"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return True
    except Exception as e:
        print(f"ERROR: Could not create directory {directory}: {str(e)}", file=sys.stderr)
        sys.exit(1)

def validate_model(model):
    """Verify model with required attributes"""
    if not isinstance(model, LinearRegression):
        print("ERROR: Model is not a LinearRegression instance", file=sys.stderr)
        sys.exit(1)
    
    required_attrs = ['coef_', 'intercept_', 'predict']
    for attr in required_attrs:
        if not hasattr(model, attr):
            print(f"ERROR: Invalid model - missing {attr}", file=sys.stderr)
            sys.exit(1)
    return True

def save_model(model, path):
    """Save model with validation"""
    try:
        validate_model(model)
        ensure_dir(os.path.dirname(path))
        joblib.dump(model, path)
        print(f"Model successfully saved to {path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save model: {str(e)}", file=sys.stderr)
        sys.exit(1)

def load_model(path):
    """Load model with validation"""
    try:
        model = joblib.load(path)
        validate_model(model)
        print(f"Model successfully loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}", file=sys.stderr)
        sys.exit(1)