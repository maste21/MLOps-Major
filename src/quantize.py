import joblib
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from src.utils import ensure_dir, load_model

def quantize_parameters(params):
    """Quantize parameters to 8-bit integers"""
    
    max_val = max(np.max(np.abs(params['coef_'])), abs(params['intercept_']))
    scale = 127 / max_val 
    
    quantized = {}
    for key, value in params.items():
        quantized[key] = np.clip(np.round(value * scale), -128, 127).astype(np.int8)
    
    return quantized, scale

def dequantize_parameters(quantized_params, scale):
    """Dequantize parameters back to floats"""
    dequantized = {}
    for key, value in quantized_params.items():
        dequantized[key] = value.astype(np.float64) / scale
    return dequantized

def calculate_quantization_error(original, dequantized, X_test=None, y_test=None):
    """Calculate quantization metrics"""
    errors = {}
    
    errors['coef_diff'] = original['coef_'] - dequantized['coef_']
    errors['intercept_diff'] = original['intercept_'] - dequantized['intercept_']
    
    metrics = {
        'max_coef_error': np.max(np.abs(errors['coef_diff'])),
        'mean_coef_error': np.mean(np.abs(errors['coef_diff'])),
        'intercept_error': np.abs(errors['intercept_diff']),
    }
    
    return metrics

def run_quantization():
    """Run full quantization pipeline"""
    try:
        ensure_dir("models")
        
        model = load_model("models/linear_regression.joblib")
        params = {
            'coef_': model.coef_,
            'intercept_': model.intercept_
        }
        
        joblib.dump(params, "models/unquant_params.joblib")
        print("Saved original parameters to models/unquant_params.joblib")
        
        quantized, scale = quantize_parameters(params)
        joblib.dump({
            'quantized': quantized,
            'scale': scale
        }, "models/quant_params.joblib")
        
        print(f"\nQuantization scale factor: {scale:.4f}")
        print("Quantized ranges:")
        print(f"  coef_: [{quantized['coef_'].min()}, {quantized['coef_'].max()}]")
        print(f"  intercept_: {quantized['intercept_']}")
        
        dequantized = dequantize_parameters(quantized, scale)
        metrics = calculate_quantization_error(params, dequantized)
        
        print("\nQuantization Errors:")
        print("="*40)
        print(f"{'Max Coefficient Error':<25}: {metrics['max_coef_error']:.8f}")
        print(f"{'Mean Coefficient Error':<25}: {metrics['mean_coef_error']:.8f}")
        print(f"{'Intercept Error':<25}: {metrics['intercept_error']:.8f}")
        
        dequant_model = LinearRegression()
        dequant_model.coef_ = dequantized['coef_']
        dequant_model.intercept_ = dequantized['intercept_']
        joblib.dump(dequant_model, "models/dequant_model.joblib")
        
        print("\nQuantization artifacts saved:")
        print("- models/unquant_params.joblib (original)")
        print("- models/quant_params.joblib (quantized)")
        print("- models/dequant_model.joblib (dequantized)")
        
    except Exception as e:
        print(f"\nERROR in quantization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_quantization()