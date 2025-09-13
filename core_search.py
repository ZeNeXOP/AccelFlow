"""
Module 3: The Simple Search Loop

This module is the orchestrator for the "dumb" pipeline. It defines a search 
space of possible hardware configurations and iterates through them. For each 
configuration, it uses the simple predictor (Module 2) to evaluate its 
performance. Finally, it filters and ranks these configurations based on 
user-defined constraints to find the best options.
"""

# Import the predictor function from the previous module
from core_predictor import predict_performance_simple

def run_simple_search(model_json: dict, constraints: dict) -> list:
    """
    Performs a brute-force search over a hardcoded hardware configuration space.

    Args:
        model_json (dict): A dictionary containing the parsed model's 
                           characteristics (from core_parser.py).
        
        constraints (dict): A dictionary with user-defined performance limits.
                            Example: {'max_latency_ms': 50, 'max_power_w': 5}

    Returns:
        list: A sorted list of dictionaries, where each dictionary represents
              a valid hardware configuration and its predicted performance.
              Returns the top 3 best configurations based on latency.
    """
    print("Starting simple search...")

    # 1. Hardcode a list of hardware configs to test (the search space)
    # In a more advanced version, this could be generated dynamically.
    hardware_search_space = []
    array_sizes = [4, 8, 12, 16]
    precisions = ['INT8', 'FP16']
    clock_speeds_ghz = [1.0, 1.2, 1.5]

    for size in array_sizes:
        for prec in precisions:
            for clock in clock_speeds_ghz:
                hardware_search_space.append({
                    'array_size': size,
                    'precision': prec,
                    'clock_ghz': clock
                })

    print(f"Generated {len(hardware_search_space)} hardware configurations to test.")

    # 2. Loop through each config, predict performance, and store valid results
    valid_configs = []
    for config in hardware_search_space:
        # Call the predictor from Module 2
        performance = predict_performance_simple(model_json, config)
        
        # Check if the predicted performance meets the user's constraints
        meets_latency = performance['latency_ms'] <= constraints['max_latency_ms']
        meets_power = performance['power_w'] <= constraints['max_power_w']

        if meets_latency and meets_power:
            # If it's a valid solution, add performance metrics to the config dict
            result = config.copy() # Avoid modifying the original list item
            result.update(performance)
            valid_configs.append(result)

    print(f"Found {len(valid_configs)} configurations that meet the constraints.")

    # 3. Sort the valid results to find the "best" ones.
    # We'll sort by latency (lower is better) as the primary metric.
    # A secondary sort by power could also be added.
    sorted_configs = sorted(valid_configs, key=lambda x: x['latency_ms'])

    # Return the top 3 best-performing configurations
    return sorted_configs[:3]

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    # Mock output from core_parser.py (Module 1)
    mock_model_json = {
      "model_name": "MobileNetV2",
      "total_flops": 315e6, # 315 Million FLOPs
      "total_params": 3.5e6,
    }

    # Mock constraints from the Streamlit UI
    mock_constraints = {
        'max_latency_ms': 2.0, # A tight constraint to test filtering
        'max_power_w': 5.0
    }

    # Run the search
    top_configurations = run_simple_search(mock_model_json, mock_constraints)

    print(f"\n--- Testing Module 3: Simple Search ---")
    print(f"Model: {mock_model_json['model_name']}")
    print(f"Constraints: {mock_constraints}")
    print("\nTop 3 Recommended Configurations:")
    
    if top_configurations:
        for i, config in enumerate(top_configurations):
            print(f"  {i+1}. Array: {config['array_size']}x{config['array_size']}, "
                  f"Precision: {config['precision']}, "
                  f"Clock: {config['clock_ghz']}GHz -> "
                  f"Latency: {config['latency_ms']}ms, "
                  f"Power: {config['power_w']}W")
    else:
        print("  No configurations found that meet the specified constraints.")
