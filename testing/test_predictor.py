# Quick test for AI predictions
from core_predictor import predict_performance_ai, predict_performance_simple
   
model_json = {
   'total_flops': 1e9,
   'total_params': 1e6,
   'flops_per_param': 1000,
   'compute_intensity': 10
}
   
hardware_config = {
   'array_size': 16,
   'precision': 'INT8', 
   'clock_ghz': 1.0,
   'batch_size': 1
}
   
print("Simple prediction:", predict_performance_simple(model_json, hardware_config))
print("AI prediction:", predict_performance_ai(model_json, hardware_config))


