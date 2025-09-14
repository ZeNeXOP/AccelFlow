import os
from core_parser import parse_onnx
from core_search import run_simple_search

def run_full_pipeline(onnx_model_path: str, constraints: dict) -> list:
    """
    End-to-end pipeline: Parse ONNX → Predict performance → Search and rank configs.
    
    Args:
        onnx_model_path (str): Path to the ONNX model file.
        constraints (dict): User constraints, e.g., {'max_latency_ms': 50, 'max_power_w': 5}.
    
    Returns:
        list: Top ranked hardware configurations that meet constraints.
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_model_path}")
    
    # Module 1: Parse the ONNX model
    model_json = parse_onnx(onnx_model_path)
    
    # Modules 2 & 3: Run search (which calls predictor internally)
    from core_search import run_hybrid_search
    top_configs = run_hybrid_search(model_json, constraints)
    
    # Module 4: Generate Verilog code
    if top_configs:
        from core_rtl import generate_rtl
        rtl_code = generate_rtl(top_configs[0])
        with open('generated_rtl.v', 'w') as f:
            f.write(rtl_code)
        print("Generated RTL saved to generated_rtl.v")
    
    return top_configs

# Example usage (for quick testing)
if __name__ == '__main__':
    example_path = 'models_onnx/mobilenetv2-7.onnx'  # Adjust to your real file
    example_constraints = {'max_latency_ms': 500, 'max_power_w': 50, max_memory_mb: 1024}
    print(run_full_pipeline(example_path, example_constraints))