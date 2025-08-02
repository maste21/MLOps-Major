from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
from src.utils import ensure_dir, save_model

def load_data(test_size=0.2, random_state=42):
    """Load and split California Housing dataset with validation"""
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    assert X.shape[1] == 8, "Dataset should have 8 features"
    assert len(X) == len(y), "X and y should have same length"
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model():
    """Train and evaluate Linear Regression model"""
    print("\nCreating LinearRegression model...")
    X_train, X_test, y_train, y_test = load_data()
    
    model = LinearRegression()
    print("Training model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print("="*40)
    print(f"{'R² Score':<20}: {r2:.6f}")
    print(f"{'Mean Squared Error':<20}: {mse:.6f}")
    print("="*40)
    
    ensure_dir("models")
    model_path = "models/linear_regression.joblib"
    save_model(model, model_path)
    
    return model, r2, mse

if __name__ == "__main__":
    model, r2_score, mse = train_model()
    print(f"\nTraining completed. Model saved to models/linear_regression.joblib")