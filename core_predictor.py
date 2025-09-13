# core_predictor.py

import joblib
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# Define the path to the directory where trained models are saved.
# This makes the path relative to the script's location, which is robust.
# Corresponds to /trained_models in your project structure. [cite: 10, 115]
MODELS_DIR = Path(__file__).parent / 'trained_models'


# ==============================================================================
#  MODULE 2: The "Dumb" Performance Predictor (Phase 1 Safety Net)
# ==============================================================================

def predict_performance_simple(model_json: dict, hardware_config: dict) -> dict:
    """
    Estimates hardware performance metrics using simple, deterministic formulas.
    This serves as the baseline predictor for the initial end-to-end pipeline.

    Args:
        model_json (dict): A dictionary containing model statistics from the parser,
                           e.g., {'total_flops': 300e6, 'total_params': 3.5e6}.
        hardware_config (dict): A dictionary describing the target hardware,
                                e.g., {'array_size': 16, 'precision': 'INT8',
                                'clock_ghz': 1.0, 'batch_size': 1}.

    Returns:
        dict: A dictionary with predicted performance metrics,
              e.g., {'latency_ms': 25.5, 'power_w': 4.1, ...}.
    """
    # --- Extract parameters ---
    # ... existing code ...
    total_flops = model_json.get('total_flops', 1e9)
    total_params = model_json.get('total_params', 1e6)

    # Handle zero-FLOPs case (e.g., empty model) as infinite latency
    if total_flops == 0:
        return {'latency_ms': float('inf'), 'power_w': 0, 'memory_mb': 0, 'throughput_ops_s': 0}

    # ... existing code ...
    array_size = hardware_config.get('array_size', 16)
    clock_ghz = hardware_config.get('clock_ghz', 1.0)
    precision = hardware_config.get('precision', 'INT8')
    batch_size = hardware_config.get('batch_size', 1)

    # --- Precision scaling factor ---
    # A simple model where lower precision is faster.
    # FP32=1.0, FP16=1.5x, INT8=2.0x
    precision_multiplier = {'FP32': 1.0, 'FP16': 1.5, 'INT8': 2.0}.get(precision, 1.0)
    bytes_per_param = {'FP32': 4, 'FP16': 2, 'INT8': 1}.get(precision, 4)

    # --- Apply formulas from the specification ---
    # Calculate the theoretical peak operations per second of the systolic array. [cite: 55]
    # (array_size * array_size) is the number of MAC units.
    # clock_ghz * 1e9 converts GHz to Hz.
    # * 2 because a MAC operation is 2 FLOPs (multiply, accumulate).
    effective_ops_per_second = (array_size * array_size) * clock_ghz * 1e9 * 2 * precision_multiplier

    if effective_ops_per_second == 0:
        return {'latency_ms': float('inf'), 'power_w': 0, 'memory_mb': 0, 'throughput_ops_s': 0}

    # Predict latency based on total operations and hardware speed. [cite: 56]
    # Multiply by 1000 to convert seconds to milliseconds.
    predicted_latency_s = (total_flops * batch_size) / effective_ops_per_second
    predicted_latency_ms = predicted_latency_s * 1000

    # Placeholder formula for power: 1 Watt per GFLOP. [cite: 57]
    predicted_power_w = total_flops * 1e-9

    # Placeholder for memory: sum of parameter memory and a fixed 50MB buffer.
    predicted_memory_mb = (total_params * bytes_per_param) / (1024 * 1024) + 50 # Base buffer

    # Throughput calculation
    predicted_throughput_ops_s = (total_flops * batch_size) / predicted_latency_s if predicted_latency_s > 0 else 0

    return {
        'latency_ms': round(predicted_latency_ms, 2),
        'power_w': round(predicted_power_w, 2),
        'memory_mb': round(predicted_memory_mb, 2),
        'throughput_ops_s': int(predicted_throughput_ops_s)
    }


# ==============================================================================
#  MODULE 7: The "Smart" AI-Powered Predictor (Phase 3 Upgrade)
# ==============================================================================

def _prepare_feature_vector(model_json: dict, hardware_config: dict) -> np.ndarray:
    """
    Converts model and hardware dictionaries into a flat numpy array for the AI model.
    This MUST match the feature format used during training in `train_ai.py`.
    """
    # Map precision string to a numerical value
    precision_map = {'FP32': 32, 'FP16': 16, 'INT8': 8}
    
    # The order of features is critical and must be consistent with the training script.
    features = [
        model_json.get('total_flops', 0),
        model_json.get('total_params', 0),
        model_json.get('sparsity_ratio', 0), # Added from main spec
        hardware_config.get('array_size', 16),
        precision_map.get(hardware_config.get('precision', 'INT8'), 8),
        hardware_config.get('batch_size', 1),
        hardware_config.get('clock_ghz', 1.0)
        # Add other features like 'rnn_unroll' if you use them in training
    ]
    return np.array(features).reshape(1, -1)


def predict_performance_ai(model_json: dict, hardware_config: dict) -> dict:
    """
    Predicts hardware performance by loading and using a pre-trained
    Mixture of Experts (MoE) model.

    This function locates the saved gating and expert models, prepares the
    input feature vector, routes it to the correct experts, and returns
    the predictions.

    Args:
        model_json (dict): Dictionary of model statistics from the parser.
        hardware_config (dict): Dictionary describing the target hardware.

    Returns:
        dict: A dictionary with AI-predicted performance metrics.
    """
    try:
        # Load the trained MoE models (gating and experts) [cite: 115, 122]
        # Assumes models are saved with these names in the 'trained_models' directory.
        gating_model = joblib.load(os.path.join(MODELS_DIR, 'gating_model.pkl'))
        expert_latency = joblib.load(os.path.join(MODELS_DIR, 'expert_latency.pkl'))
        expert_power = joblib.load(os.path.join(MODELS_DIR, 'expert_power.pkl'))
        expert_memory = joblib.load(os.path.join(MODELS_DIR, 'expert_memory.pkl'))
        expert_throughput = joblib.load(os.path.join(MODELS_DIR, 'expert_throughput.pkl'))

        experts = {
            'latency': expert_latency,
            'power': expert_power,
            'memory': expert_memory,
            'throughput': expert_throughput
        }

    except FileNotFoundError:
        print("---")
        print("ERROR: AI model files not found. Ensure `train_ai.py` has been run.")
        print(f"Looked in: {MODELS_DIR}")
        print("Falling back to simple predictor.")
        print("---")
        return predict_performance_simple(model_json, hardware_config)

    # Prepare the feature vector from the inputs [cite: 111]
    feature_vector = _prepare_feature_vector(model_json, hardware_config)

    # --- MoE Gating & Prediction ---
    # 1. Use the gating model to decide which expert to use for each prediction.
    #    (For a simple hackathon implementation, we can assume the gating model
    #    tells us which expert is 'best' overall, or we can have one expert per target)
    #    Your spec calls for 4 experts, implying one per metric. We will predict with each.

    latency_pred = experts['latency'].predict(feature_vector)[0]
    power_pred = experts['power'].predict(feature_vector)[0]
    memory_pred = experts['memory'].predict(feature_vector)[0]
    throughput_pred = experts['throughput'].predict(feature_vector)[0]

    return {
        'latency_ms': round(latency_pred, 2),
        'power_w': round(power_pred, 2),
        'memory_mb': round(memory_pred, 2),
        'throughput_ops_s': int(throughput_pred)
    }


# ==============================================================================
#  EXAMPLE USAGE (for testing this module directly)
# ==============================================================================

if __name__ == '__main__':
    print("--- Testing core_predictor.py ---")

    # 1. Define dummy inputs, mimicking outputs from the parser and UI
    mock_model_json = {
        'model_name': 'TestNetV2',
        'total_flops': 300e6,
        'total_params': 3.5e6,
        'sparsity_ratio': 0.1,
        'layers': []
    }

    mock_hardware_config = {
        'array_size': 16,
        'precision': 'INT8',
        'clock_ghz': 1.2,
        'batch_size': 4
    }

    # 2. Test the "dumb" formula-based predictor
    print("\n[1] Testing Simple Predictor (Module 2)...")
    simple_predictions = predict_performance_simple(mock_model_json, mock_hardware_config)
    print(f"   > Simple Prediction Results: {simple_predictions}")

    # 3. Test the "smart" AI-based predictor
    #    NOTE: This will fail until you run `train_ai.py` to create the .pkl model files.
    #    To simulate, we will create dummy model files.
    print("\n[2] Testing AI Predictor (Module 7)...")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Create dummy models for testing purposes if they don't exist
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.dummy import DummyClassifier
        
        # This is just to allow the script to run without the actual training script.
        # The real models will be generated by train_ai.py
        if not os.path.exists(os.path.join(MODELS_DIR, 'gating_model.pkl')):
            print("   > NOTE: Creating dummy AI models for testing...")
            dummy_X = np.random.rand(10, 7)
            dummy_y = np.random.rand(10)
            
            # Dummy Gating
            gating = DummyClassifier(strategy="most_frequent")
            gating.fit(dummy_X, [0,1,0,1,0,1,0,1,0,1])
            joblib.dump(gating, os.path.join(MODELS_DIR, 'gating_model.pkl'))

            # Dummy Experts
            expert = RandomForestRegressor(n_estimators=5)
            expert.fit(dummy_X, dummy_y)
            joblib.dump(expert, os.path.join(MODELS_DIR, 'expert_latency.pkl'))
            joblib.dump(expert, os.path.join(MODELS_DIR, 'expert_power.pkl'))
            joblib.dump(expert, os.path.join(MODELS_DIR, 'expert_memory.pkl'))
            joblib.dump(expert, os.path.join(MODELS_DIR, 'expert_throughput.pkl'))

        ai_predictions = predict_performance_ai(mock_model_json, mock_hardware_config)
        print(f"   > AI Prediction Results: {ai_predictions}")

    except ImportError:
        print("   > Scikit-learn not found. Cannot create dummy models for AI test.")
    except Exception as e:
        print(f"   > An error occurred during AI prediction test: {e}")