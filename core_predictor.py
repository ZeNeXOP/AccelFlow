# core_predictor.py

import joblib
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn

# --- Configuration ---
# Define the path to the directory where trained models are saved.
# This makes the path relative to the script's location, which is robust.
# Corresponds to /trained_models in your project structure. [cite: 10, 115]
MODELS_DIR = Path(__file__).parent / 'trained_models'

# Setup device for AI predictions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # Precision affects both computational throughput and memory footprint
    if precision == 'INT8':
        precision_factor = 1.0  # Baseline
        memory_factor = 1.0
        power_factor = 1.0
    elif precision == 'FP16':
        precision_factor = 0.5  # Half the throughput of INT8
        memory_factor = 2.0     # Twice the memory
        power_factor = 1.5      # More power consumption
    elif precision == 'FP32':
        precision_factor = 0.25 # Quarter the throughput of INT8
        memory_factor = 4.0     # Four times the memory
        power_factor = 2.0      # Even more power consumption
    else:
        precision_factor = 1.0  # Default to INT8 behavior
        memory_factor = 1.0
        power_factor = 1.0

    # --- Performance Calculations ---
    # 1. Latency (milliseconds)
    # Basic formula: Total operations / (compute capability * clock speed)
    # Compute capability: array_size^2 * operations_per_cycle * precision_factor
    operations_per_cycle = array_size * array_size * 2  # 2 ops per MAC (multiply + accumulate)
    compute_capability = operations_per_cycle * precision_factor
    cycles_required = total_flops / compute_capability
    time_seconds = cycles_required / (clock_ghz * 1e9)  # Convert GHz to Hz
    latency_ms = time_seconds * 1000 / batch_size  # Account for batch processing

    # 2. Power Consumption (Watts)
    # Static power + dynamic power * activity factor
    base_power = 2.0  # Base power for the system
    dynamic_power_per_mac = 0.000001  # 1uW per MAC operation (simplified)
    total_mac_ops = total_flops / 2  # FLOPs = 2 * MAC operations
    power_w = base_power + (dynamic_power_per_mac * total_mac_ops * power_factor / batch_size)

    # 3. Memory Requirements (MB)
    # Model parameters + activations + intermediate results
    param_memory = total_params * memory_factor * 4 / 1e6  # 4 bytes per parameter (conservative)
    activation_memory = total_flops * 0.1 / 1e6  # Simplified: 0.1 bytes per FLOP for activations
    memory_mb = param_memory + activation_memory

    # 4. Throughput (Operations per second)
    throughput_ops_s = total_flops / (time_seconds + 1e-9)  # Avoid division by zero

    return {
        'latency_ms': max(0, latency_ms),
        'power_w': max(0, power_w),
        'memory_mb': max(0, memory_mb),
        'throughput_ops_s': max(0, throughput_ops_s)
    }

# ==============================================================================
#  ENHANCED AI PREDICTOR (Trained MoE Model)
# ==============================================================================


# ... replace the Expert and MoEModel classes with this exact code ...
class Expert(nn.Module):
    """Expert with 32 output features to match trained model architecture"""
    def __init__(self, input_size, output_size=32, hidden_size=128, dropout_rate=0.2, positive_output=True):
        super(Expert, self).__init__()
        self.positive_output = positive_output
        
        # 13-layer architecture that matches the trained model
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),          # 0
            nn.BatchNorm1d(hidden_size),                 # 1
            nn.ReLU(),                                   # 2
            nn.Dropout(dropout_rate),                    # 3
            nn.Linear(hidden_size, hidden_size // 2),    # 4: 128 -> 64
            nn.BatchNorm1d(hidden_size // 2),            # 5
            nn.ReLU(),                                   # 6
            nn.Dropout(dropout_rate),                    # 7
            nn.Linear(hidden_size // 2, hidden_size // 4), # 8: 64 -> 32
            nn.BatchNorm1d(hidden_size // 4),            # 9
            nn.ReLU(),                                   # 10
            nn.Dropout(dropout_rate),                    # 11
            nn.Linear(hidden_size // 4, output_size)     # 12: 32 -> 32
        )
 
    def forward(self, x):
        output = self.net(x)
        if self.positive_output:
            return nn.ReLU()(output)
        return output


class CompatibleMoEModel(nn.Module):
    """MoE model that matches the architecture of the trained model"""
    def __init__(self, input_size, num_experts=6, targets=None):  # Note: 6 experts!
        super(CompatibleMoEModel, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.targets = targets or ['latency_ms', 'power_w', 'memory_mb', 'throughput_ops_s']
        
        # Enhanced gating network (matches trained architecture)
        self.gating = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts * len(self.targets)),
            nn.Softmax(dim=1)
        )
        
        # Create experts with 32 output features (matches trained model)
        self.experts = nn.ModuleDict()
        for target in self.targets:
            positive_output = target in ['latency_ms', 'power_w', 'memory_mb']
            self.experts[target] = nn.ModuleList([
                Expert(input_size, output_size=32, positive_output=positive_output) 
                for _ in range(num_experts)
            ])

    def forward(self, x):
        # Same forward logic but with 32-feature experts
        gate_outputs = self.gating(x).view(x.size(0), len(self.targets), self.num_experts)
        
        predictions = {}
        for i, target in enumerate(self.targets):
            expert_outputs = []
            for expert in self.experts[target]:
                expert_outputs.append(expert(x))
            
            expert_outputs = torch.stack(expert_outputs, dim=1)
            gate_weights = gate_outputs[:, i, :].unsqueeze(-1)
            
            # Take only the first output feature (original design was 1 output)
            predictions[target] = (expert_outputs * gate_weights).sum(dim=1)[:, :1]
        
        return predictions
    
    
def load_trained_moe():
    """Load the trained MoE model for inference with architecture compatibility"""
    try:
        # Load metadata
        metadata = joblib.load(MODELS_DIR / 'model_metadata.pkl')
        
        # Create model with the correct architecture that matches the trained weights
        model = CompatibleMoEModel(
            input_size=metadata['input_size'],
            num_experts=metadata['num_experts'],
            targets=metadata['targets']
        )
        
        # Load state dict with strict=False to ignore architecture mismatches
        state_dict = torch.load(MODELS_DIR / 'enhanced_moe_model.pth', map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(DEVICE)
        model.eval()
        
        # Load scalers
        feature_scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
        target_scaler = joblib.load(MODELS_DIR / 'target_scaler.pkl')
        
        return model, feature_scaler, target_scaler, metadata
        
    except FileNotFoundError:
        print("ERROR: Trained model not found. Please train the model first.")
        return None, None, None, None

   
def predict_performance_ai(model_json: dict, hardware_config: dict) -> dict:
    """AI-powered performance prediction using trained MoE model"""
    # Use the compatible model instead of original
    model, feature_scaler, target_scaler, metadata = load_trained_moe()
    if model is None:
        # Fall back to simple predictor if AI model not available
        return predict_performance_simple(model_json, hardware_config)
    
    # Prepare features
    features = [
        model_json.get('total_flops', 0),
        model_json.get('total_params', 0),
        hardware_config.get('array_size', 8),
        0 if hardware_config.get('precision', 'FP16') == 'INT8' else 
        1 if hardware_config.get('precision', 'FP16') == 'FP16' else 2,
        hardware_config.get('batch_size', 1),
        hardware_config.get('clock_ghz', 1.0),
        model_json.get('flops_per_param', 0),
        model_json.get('compute_intensity', 0),
        hardware_config.get('array_utilization', 0.5),
        hardware_config.get('power_efficiency', 1.0),
        hardware_config.get('memory_efficiency', 1.0),
    ]
    
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features_scaled = feature_scaler.transform(features)
    
    # Predict
    with torch.no_grad():
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        predictions = model(features_tensor)
    
    # Convert and unscale
    pred_array = np.zeros((1, len(metadata['targets'])))
    for i, target in enumerate(metadata['targets']):
        pred_array[0, i] = predictions[target].cpu().numpy()
    
    pred_unscaled = target_scaler.inverse_transform(pred_array)
    
    # Format results
    results = {}
    for i, target in enumerate(metadata['targets']):
        results[target] = max(0, pred_unscaled[0, i])  # Ensure non-negative
    
    return results

# Unified predictor function that automatically chooses the best available method
def predict_performance(model_json: dict, hardware_config: dict, use_ai: bool = True) -> dict:
    """
    Unified performance predictor - uses AI if available, falls back to simple formulas
    
    Args:
        model_json: Model characteristics from parser
        hardware_config: Hardware configuration to evaluate
        use_ai: Whether to attempt using AI prediction (default: True)
    
    Returns:
        Performance metrics dictionary
    """
    if use_ai:
        try:
            return predict_performance_ai(model_json, hardware_config)
        except:
            # Fall back to simple predictor if AI fails
            return predict_performance_simple(model_json, hardware_config)
    else:
        return predict_performance_simple(model_json, hardware_config)