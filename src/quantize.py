import joblib
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from src.utils import ensure_dir

def quantize_parameters(params, scale=100):
    """Quantize parameters to 8-bit integers with detailed logging"""
    print("\nStarting quantization process...")
    print(f"Using scale factor: {scale}")
    
    quantized = {}
    for key, value in params.items():
        print(f"\nParameter: {key}")
        print(f"Original dtype: {value.dtype if hasattr(value, 'dtype') else type(value)}")
        print(f"Original range: [{np.min(value) if hasattr(value, 'min') else value}, "
              f"{np.max(value) if hasattr(value, 'max') else value}]")
        
        if isinstance(value, np.ndarray):
            quantized[key] = np.round(value * scale).astype(np.uint8)
        else:
            quantized[key] = np.round(value * scale).astype(np.uint8)
            
        print(f"Quantized range: [{np.min(quantized[key])}, {np.max(quantized[key])}]")
    
    return quantized, scale  

def run_quantization():
    """Run full quantization pipeline with enhanced output"""
    try:
        ensure_dir("models")
        
        try:
            model = joblib.load("models/linear_regression.joblib")
        except FileNotFoundError:
            print("ERROR: Model not found. Run train.py first.", file=sys.stderr)
            sys.exit(1)
            
        params = {
            'coef_': model.coef_,
            'intercept_': model.intercept_
        }
        
        joblib.dump(params, "models/unquant_params.joblib")
        print("\nSaved original parameters to models/unquant_params.joblib")
        
        quantized, scale = quantize_parameters(params)
        joblib.dump(quantized, "models/quant_params.joblib")
        print("\nSaved quantized parameters to models/quant_params.joblib")
        
        dequantized = {
            'coef_': quantized['coef_'].astype(np.float64) / scale,
            'intercept_': quantized['intercept_'].astype(np.float64) / scale
        }
        
        print("\nQuantization Results:")
        print("="*40)
        print(f"{'Metric':<25} | {'Original':>12} | {'Quantized':>12}")
        print("-"*40)
        for key in params:
            if isinstance(params[key], np.ndarray):
                orig_val = str(params[key][:3].round(6))
                dequant_val = str(dequantized[key][:3].round(6))
            else:
                orig_val = f"{params[key]:.6f}"
                dequant_val = f"{dequantized[key]:.6f}"
            print(f"{key:<25} | {orig_val:>12} | {dequant_val:>12}")
        
        coef_error = np.max(np.abs(params['coef_'] - dequantized['coef_']))
        intercept_error = abs(params['intercept_'] - dequantized['intercept_'])
        
        print("\nQuantization Errors:")
        print(f"Max coefficient error: {coef_error:.8f}")
        print(f"Intercept error: {intercept_error:.8f}")
        print(f"Average error: {(coef_error + intercept_error)/2:.8f}")
        
    except Exception as e:
        print(f"ERROR during quantization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_quantization()