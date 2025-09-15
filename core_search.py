"""
Enhanced Neural Architecture Search for AccelForge

This module provides:
1. run_simple_search() - Original brute-force search (Phase 1)
2. run_rl_nas_search() - RL-enhanced intelligent search (Phase 2)  
3. run_hybrid_search() - Combined approach for optimal results
4. run_enhanced_search() - Backward compatibility wrapper

Features:
- Multi-objective optimization (latency, power, throughput)
- Constraint-aware exploration
- Intelligent mutation of successful configurations
- Hybrid brute-force + RL approach
"""

import random
import numpy as np
from core_predictor import predict_performance, predict_performance_simple

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

    # --- Define the search space ---
    # These represent different hardware configurations to test
    array_sizes = [4, 8, 12, 16, 24, 32]
    precisions = ['INT8', 'FP16', 'FP32']
    clock_speeds = [round(x, 2) for x in np.arange(0.8, 2.6, 0.2)]  # 0.8 to 2.4 GHz
    batch_sizes = [1, 2, 4, 8, 16]
    
    # Generate all combinations (brute force)
    all_configs = []
    for array_size in array_sizes:
        for precision in precisions:
            for clock_ghz in clock_speeds:
                for batch_size in batch_sizes:
                    config = {
                        'array_size': array_size,
                        'precision': precision,
                        'clock_ghz': clock_ghz,
                        'batch_size': batch_size
                    }
                    all_configs.append(config)
    
    print(f"Generated {len(all_configs)} hardware configurations to test.")
    
    # --- Evaluate each configuration ---
    valid_configs = []
    for config in all_configs:
        # Get performance prediction
        performance = predict_performance_simple(model_json, config)
        
        # Check if this configuration meets all constraints
        meets_constraints = True
        if 'max_latency_ms' in constraints and performance['latency_ms'] > constraints['max_latency_ms']:
            meets_constraints = False
        if 'max_power_w' in constraints and performance['power_w'] > constraints['max_power_w']:
            meets_constraints = False
        if 'max_memory_mb' in constraints and performance['memory_mb'] > constraints['max_memory_mb']:
            meets_constraints = False
        
        if meets_constraints:
            # Combine config with performance data
            result = config.copy()
            result.update(performance)
            valid_configs.append(result)

    print(f"Found {len(valid_configs)} configurations that meet the constraints.")

    # --- Sort and return top results ---
    # Sort by latency (primary optimization goal)
    valid_configs.sort(key=lambda x: x['latency_ms'])
    
    # Return top 3 configurations
    return valid_configs[:3]

def run_rl_nas_search(model_json: dict, constraints: dict, episodes: int = 50) -> list:
    """
    Simple RL-based Neural Architecture Search using intelligent exploration.
    This enhances the existing brute-force search with policy gradient concepts.
    """
    print("Starting RL-enhanced NAS search...")
    
    # Define search space (same as your existing one)
    search_space = {
        'array_size': [4, 8, 12, 16, 24, 32],
        'precision': ['INT8', 'FP16', 'FP32'],
        'clock_ghz': [round(x, 2) for x in np.arange(0.8, 2.6, 0.2)],
        'batch_size': [1, 2, 4, 8, 16]
    }
    
    best_configs = []
    best_reward = -float('inf')
    explored_configs = set()
    
    for episode in range(episodes):
        # Intelligent sampling: prefer configurations similar to previous good ones
        if best_configs and random.random() < 0.7:  # 70% exploitation, 30% exploration
            # Mutate a previously good configuration
            base_config = random.choice(best_configs).copy()
            config = mutate_config(base_config, search_space, mutation_rate=0.3)
        else:
            # Completely new random configuration
            config = {}
            for param, values in search_space.items():
                config[param] = random.choice(values)
        
        # Skip duplicates
        config_key = tuple(sorted((k, v) for k, v in config.items()))
        if config_key in explored_configs:
            continue
        explored_configs.add(config_key)
        
        # Evaluate with AI predictor if available, fall back to simple
        performance = predict_performance(model_json, config, use_ai=True)
        
        # Calculate multi-objective reward
        reward = calculate_rl_reward(performance, constraints)
        
        # Track best configurations
        if reward > best_reward:
             best_reward = reward
        best_result = config.copy()
        best_result.update(performance)
        best_configs.append(best_result)
        
        # Keep only top 5 based on reward
        best_configs.sort(key=lambda x: calculate_rl_reward(x, constraints), reverse=True)
        best_configs = best_configs[:5]
        
    # Print progress every 10 episodes
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}/{episodes}, Best Reward: {best_reward:.4f}")

    return best_configs

def mutate_config(config, search_space, mutation_rate=0.3):
    """Mutate a configuration by randomly changing parameters"""
    new_config = config.copy()
    for param in search_space.keys():
        if random.random() < mutation_rate:
            new_config[param] = random.choice(search_space[param])
    return new_config

def calculate_rl_reward(performance, constraints):
    """Calculate reward for RL-based search (multi-objective optimization)"""
    latency = performance.get('latency_ms', float('inf'))
    power = performance.get('power_w', float('inf'))
    memory = performance.get('memory_mb', float('inf'))
    throughput = performance.get('throughput_ops_s', 0)
    
    # Check constraints (hard constraints) - return -inf for violations
    if (latency > constraints.get('max_latency_ms', float('inf')) or 
        power > constraints.get('max_power_w', float('inf')) or 
        memory > constraints.get('max_memory_mb', float('inf'))):
        return -float('inf')  # Absolute penalty for constraint violation
    
    # Multi-objective reward function for valid configurations
    latency_reward = 1.0 / (latency + 1e-6)
    throughput_reward = throughput / 1e9  # Scale for numerical stability
    power_reward = 1.0 / (power + 1e-6)
    
    # Weighted combination of objectives
    return 0.4 * latency_reward + 0.3 * throughput_reward + 0.3 * power_reward

def run_rl_nas_search(model_json: dict, constraints: dict, episodes: int = 50) -> list:
    """Simple RL-based Neural Architecture Search using intelligent exploration."""
    print("Starting RL-enhanced NAS search...")
    
    search_space = {
        'array_size': [4, 8, 12, 16, 24, 32],
        'precision': ['INT8', 'FP16', 'FP32'],
        'clock_ghz': [round(x, 2) for x in np.arange(0.8, 2.6, 0.2)],
        'batch_size': [1, 2, 4, 8, 16]
    }
    
    best_configs = []
    best_reward = -float('inf')
    explored_configs = set()
    
    for episode in range(episodes):
        # Intelligent sampling: prefer configurations similar to previous good ones
        if best_configs and random.random() < 0.7:  # 70% exploitation, 30% exploration
            # Mutate a previously good configuration
            base_config = random.choice(best_configs).copy()
            current_config = mutate_config(base_config, search_space, mutation_rate=0.3)  # CHANGED: config -> current_config
        else:
            # Completely new random configuration
            current_config = {}  # CHANGED: config -> current_config
            for param, values in search_space.items():
                current_config[param] = random.choice(values)  # CHANGED: config -> current_config
        
        # Skip duplicates
        config_key = tuple(sorted((k, v) for k, v in current_config.items()))  # CHANGED: config -> current_config
        if config_key in explored_configs:
            continue
        explored_configs.add(config_key)
        
        # Evaluate with AI predictor if available, fall back to simple
        performance = predict_performance(model_json, current_config, use_ai=True)  # CHANGED: config -> current_config
        
        # Calculate multi-objective reward
        reward = calculate_rl_reward(performance, constraints)
        
        # Only track configurations that meet constraints (reward > -inf)
        if reward > -float('inf'):
            best_result = current_config.copy()  # CHANGED: config -> current_config
            best_result.update(performance)
            best_configs.append(best_result)
            
            # Keep only top 5 based on reward
            best_configs.sort(key=lambda x: calculate_rl_reward(x, constraints), reverse=True)
            best_configs = best_configs[:5]
            
            # Update best reward
            if reward > best_reward:
                best_reward = reward
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            valid_configs = sum(1 for cfg in best_configs if calculate_rl_reward(cfg, constraints) > -float('inf'))
            print(f"Episode {episode+1}/{episodes}, Valid configs: {valid_configs}, Best Reward: {best_reward:.4f}")
    
    # Return only configurations that actually meet constraints
    valid_configs = [cfg for cfg in best_configs if calculate_rl_reward(cfg, constraints) > -float('inf')]
    return valid_configs
    
    # Return only configurations that actually meet constraints
    valid_configs = [cfg for cfg in best_configs if calculate_rl_reward(cfg, constraints) > -float('inf')]
    return valid_configs

def run_hybrid_search(model_json: dict, constraints: dict) -> list:
    """Enhanced search that combines brute-force and RL approaches."""
    print("Starting hybrid search (brute-force + RL)...")
    
    # Run your existing brute-force search for comprehensive coverage
    print("Phase 1: Brute-force exploration...")
    brute_force_results = run_simple_search(model_json, constraints)
    
    # Run RL-enhanced search for intelligent exploitation
    print("Phase 2: RL-based optimization...")
    rl_results = run_rl_nas_search(model_json, constraints, episodes=30)
    
    # Combine and deduplicate results
    all_results = brute_force_results + rl_results
    unique_results = []
    seen_configs = set()
    
    for result in all_results:
        config_key = tuple(sorted((k, v) for k, v in result.items() 
                                if k in ['array_size', 'precision', 'clock_ghz', 'batch_size']))
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            unique_results.append(result)
    
    # Sort by primary objective (latency) and return top results
    unique_results.sort(key=lambda x: x.get('latency_ms', float('inf')))
    
    # Filter out configurations that don't meet constraints
    valid_results = []
    for result in unique_results:
        if (result['latency_ms'] <= constraints.get('max_latency_ms', float('inf')) and
            result['power_w'] <= constraints.get('max_power_w', float('inf')) and
            result['memory_mb'] <= constraints.get('max_memory_mb', float('inf'))):
            valid_results.append(result)
    
    print(f"Hybrid search complete. Found {len(valid_results)} valid configurations.")
    return valid_results[:5]

# For backward compatibility with your existing pipeline
def run_enhanced_search(model_json: dict, constraints: dict) -> list:
    """Enhanced version of your original search function"""
    return run_hybrid_search(model_json, constraints)

# Quick test function
def test_enhanced_search():
    """Test the enhanced search functionality"""
    # Test with sample model data
    model_json = {
        'model_name': 'test_model',
        'total_flops': 2.5e9,
        'total_params': 5.2e6,
        'flops_per_param': 480.77,
        'compute_intensity': 15.2,
        'layers': []
    }
    
    constraints = {
        'max_latency_ms': 50,
        'max_power_w': 25,
        'max_memory_mb': 512
    }
    
    print("Testing enhanced search...")
    results = run_hybrid_search(model_json, constraints)
    
    print(f"Found {len(results)} optimal configurations:")
    for i, config in enumerate(results, 1):
        print(f"#{i}: {config}")
    
    return results

if __name__ == "__main__":
    test_enhanced_search()