import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse

# Define the output directory
DATA_DIR = Path(__file__).parent / 'data'

def synthesize_data(num_samples: int = 500, seed: int = None) -> pd.DataFrame:
    """Generates a synthetic dataset for training hardware performance predictors."""
    if seed:
        np.random.seed(seed)
    
    print(f"Synthesizing {num_samples} data points with seed {seed}...")
    
    data = {
        'total_flops': np.random.uniform(1e8, 80e9, num_samples),
        'total_params': np.random.uniform(1e6, 150e6, num_samples),
        'array_size': np.random.choice([4, 8, 12, 16, 24, 32], num_samples),
        'precision_val': np.random.choice([8, 16, 32], num_samples),
        'batch_size': np.random.choice([1, 2, 4, 8, 16], num_samples),
        'clock_ghz': np.random.uniform(0.8, 2.5, num_samples)
    }
    df = pd.DataFrame(data)

    # Performance Calculation with Noise
    precision_multiplier = df['precision_val'].apply(lambda x: {32: 1.0, 16: 1.5, 8: 2.0}.get(x, 1.0))
    effective_ops = (df['array_size']**2) * df['clock_ghz'] * 1e9 * 2 * precision_multiplier
    base_latency_s = (df['total_flops'] * df['batch_size']) / effective_ops
    noise = np.random.normal(1.0, 0.15, num_samples)
    df['latency_ms'] = (base_latency_s * 1000) * noise

    power_coeff = np.random.uniform(0.8e-9, 1.2e-9, num_samples)
    df['power_w'] = (df['total_flops'] * power_coeff) + (df['array_size'] * 0.1) * df['clock_ghz']

    bytes_per_param = df['precision_val'] / 8
    buffer_overhead = np.random.uniform(50, 200, num_samples)
    df['memory_mb'] = ((df['total_params'] * bytes_per_param) / (1024**2)) + buffer_overhead
    
    df['throughput_ops_s'] = (df['total_flops'] * df['batch_size']) / (df['latency_ms'] / 1000)
    
    df['data_source'] = 'synthetic' # Add a column to identify the data source

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print("Synthesis complete.")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic performance data.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate.")
    parser.add_argument("--output_file", type=str, default="synthetic_data.csv", help="Output CSV file name.")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    
    # Use the first 4 digits of the output filename hash as a seed for reproducibility
    seed = hash(args.output_file) & 0xffff 
    
    synthetic_df = synthesize_data(num_samples=args.num_samples, seed=seed)
    
    output_path = DATA_DIR / args.output_file
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}")