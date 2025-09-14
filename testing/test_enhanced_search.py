#!/usr/bin/env python3
"""
Quick test script for the enhanced RL-based NAS search
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_search import run_hybrid_search

# Test with sample model data
model_json = {
    'model_name': 'test_model',
    'total_flops': 2.5e9,        # 2.5 billion FLOPs
    'total_params': 5.2e6,       # 5.2 million parameters
    'flops_per_param': 480.77,   # FLOPs per parameter
    'compute_intensity': 15.2,   # Compute intensity
    'layers': []                 # Empty layers for test
}

# Realistic constraints
constraints = {
    'max_latency_ms': 50,        # 50ms max latency
    'max_power_w': 25,           # 25W max power
    'max_memory_mb': 512         # 512MB max memory
}

print("🧪 Testing Enhanced RL-Based NAS Search...")
print(f"Model: {model_json['model_name']}")
print(f"FLOPs: {model_json['total_flops']/1e9:.1f} GFLOPs")
print(f"Parameters: {model_json['total_params']/1e6:.1f}M")
print(f"Constraints: {constraints}")
print("-" * 50)

# Run the enhanced search
results = run_hybrid_search(model_json, constraints)

print(f"✅ Found {len(results)} optimal configurations:")
print("-" * 50)

for i, config in enumerate(results, 1):
    print(f"#{i}:")
    print(f"  Hardware: {config['array_size']}x{config['array_size']} array, "
          f"{config['precision']}, {config['clock_ghz']}GHz, "
          f"batch={config['batch_size']}")
    print(f"  Performance: {config['latency_ms']:.2f}ms latency, "
          f"{config['power_w']:.1f}W power, "
          f"{config['memory_mb']:.0f}MB memory")
    print(f"  Throughput: {config['throughput_ops_s']/1e9:.2f} GOPS/s")
    print()
