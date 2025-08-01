import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import ensure_dir

def quantize_parameters(params, scale=100):
    """Quantize parameters to 8-bit integers"""
    quantized = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            quantized[key] = np.round(value * scale).astype(np.uint8)
        else:
            quantized[key] = np.round(value * scale).astype(np.uint8)
    return quantized

def dequantize_parameters(quantized_params, scale=100):
    """Dequantize parameters back to floats"""
    dequantized = {}
    for key, value in quantized_params.items():
        if isinstance(value, np.ndarray):
            dequantized[key] = value.astype(np.float64) / scale
        else:
            dequantized[key] = np.float64(value) / scale
    return dequantized

def run_quantization():
    """Run full quantization pipeline"""
    ensure_dir("models")
    model = joblib.load("models/linear_regression.joblib")
    
    params = {
        'coef_': model.coef_,
        'intercept_': model.intercept_
    }
    
    joblib.dump(params, "models/unquant_params.joblib")
    
    quantized = quantize_parameters(params)
    joblib.dump(quantized, "models/quant_params.joblib")
    
    dequantized = dequantize_parameters(quantized)
    
    new_model = LinearRegression()
    new_model.coef_ = dequantized['coef_']
    new_model.intercept_ = dequantized['intercept_']
    
    joblib.dump(new_model, "models/dequant_model.joblib")
    print("Quantization process completed successfully")

if __name__ == "__main__":
    run_quantization()