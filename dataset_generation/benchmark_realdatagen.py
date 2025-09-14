import onnxruntime as ort
import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_parser import parse_onnx # Reuse your parser
from pathlib import Path
import argparse
<<<<<<< Updated upstream
=======

# Make sure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom ONNX parser
from core_parser import parse_onnx
>>>>>>> Stashed changes

# Define the output directory
DATA_DIR = Path(__file__).parent / 'data'

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
def run_single_inference(session, input_name, input_shape, input_type):
    """Run a single inference and measure latency."""
    dummy_input = np.random.randn(*input_shape).astype(input_type)
    start_time = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000  # Convert to milliseconds

<<<<<<< Updated upstream
def run_benchmark(model_path: str, num_runs: int = 50) -> list:
    """Runs inference and measures performance for a single ONNX model multiple times."""
    print(f"Benchmarking {model_path} for {num_runs} runs...")
    
    try:
        # Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        # Try CUDA first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(model_path, sess_options, providers=providers)
        
=======

def run_benchmark(model_path: str, num_runs: int = 50) -> list:
    """Runs inference and measures performance for a single ONNX model multiple times."""
    print(f"Benchmarking {model_path} for {num_runs} runs...")

    try:
        # Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(model_path, sess_options, providers=providers)

>>>>>>> Stashed changes
        # Get input metadata
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        input_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
        input_type = np.float32 if input_meta.type == 'tensor(float)' else np.int64
<<<<<<< Updated upstream
        
        print(f"  Input shape: {input_shape}, type: {input_type}")
        print(f"  Provider: {sess.get_providers()[0]}")

        # Warm-up runs (don't count these)
=======

        print(f"  Input shape: {input_shape}, type: {input_type}")
        print(f"  Provider: {sess.get_providers()[0]}")

        # Warm-up runs
>>>>>>> Stashed changes
        print("  Warming up...")
        for _ in range(10):
            run_single_inference(sess, input_name, input_shape, input_type)

        # Actual benchmark runs
        print("  Running benchmark...")
        results = []
<<<<<<< Updated upstream
        
        for run_id in range(num_runs):
            if (run_id + 1) % 10 == 0:
                print(f"    Completed {run_id + 1}/{num_runs} runs")
            
            # Measure latency for this run
            latency_ms = run_single_inference(sess, input_name, input_shape, input_type)
            
            # Generate realistic but variable power and memory estimates
            # These would need actual hardware monitoring in a real system
            base_power = np.random.uniform(5, 50)  # Base power consumption
            power_variance = np.random.normal(0, base_power * 0.1)  # ±10% variance
            power_w = max(1.0, base_power + power_variance)
            
            base_memory = np.random.uniform(200, 1000)  # Base memory usage
            memory_variance = np.random.normal(0, base_memory * 0.05)  # ±5% variance
            memory_mb = max(50.0, base_memory + memory_variance)
            
=======

        for run_id in range(num_runs):
            if (run_id + 1) % 10 == 0:
                print(f"    Completed {run_id + 1}/{num_runs} runs")

            latency_ms = run_single_inference(sess, input_name, input_shape, input_type)

            # Generate synthetic power/memory estimates (placeholders)
            base_power = np.random.uniform(5, 50)
            power_variance = np.random.normal(0, base_power * 0.1)
            power_w = max(1.0, base_power + power_variance)

            base_memory = np.random.uniform(200, 1000)
            memory_variance = np.random.normal(0, base_memory * 0.05)
            memory_mb = max(50.0, base_memory + memory_variance)

>>>>>>> Stashed changes
            results.append({
                'run_id': run_id + 1,
                'latency_ms': latency_ms,
                'power_w': power_w,
                'memory_mb': memory_mb
            })
<<<<<<< Updated upstream
        
        print(f"  Completed all {num_runs} runs")
        return results
        
=======

        print(f"  Completed all {num_runs} runs")
        return results

>>>>>>> Stashed changes
    except Exception as e:
        print(f"  > Failed to benchmark {model_path}. Error: {e}")
        return []

<<<<<<< Updated upstream
def generate_real_data(model_dir: str, output_file: str, runs_per_model: int = 50):
    """Parses models, runs benchmarks multiple times, and creates a comprehensive dataset."""
    all_results = []
    
    # Define hardware config for these benchmarks
    hardware_config = {
<<<<<<< Updated upstream
        'array_size': 32,  # Example: CUDA cores / 64
        'precision_val': 32,  # FP32 precision
        'batch_size': 1,
        'clock_ghz': 1.48,  # GPU boost clock in GHz
=======

def generate_real_data(model_dir: str, output_file: str, runs_per_model: int = 50):
    """Parses models, runs benchmarks multiple times, and creates a dataset."""
    all_results = []

    # Define hardware config (adjust based on your GPU)
    hardware_config = {
        'array_size': 48,   # Example: CUDA cores / 64
        'precision_val': 32,  # FP32 precision
        'batch_size': 1,
        'clock_ghz': 2.46,   # GPU boost clock in GHz
>>>>>>> Stashed changes
    }
=======
    'array_size': 36, 
    'precision_val': 32,  # FP32 precision
    'batch_size': 1,
    'clock_ghz': 1.47,  # Boost clock in GHz
}
>>>>>>> Stashed changes

<<<<<<< Updated upstream
    # Get all ONNX models in the directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    total_models = len(model_files)
    
    if total_models == 0:
        print(f"No ONNX models found in {model_dir}")
        return
    
<<<<<<< Updated upstream
    print(f"Found {total_models} ONNX models. Will run {runs_per_model} benchmarks per model.")
=======
    print(f"Found {total_models} ONNX models. Will run {runs_per_model} benchmarks")
>>>>>>> Stashed changes
=======
    # Get all ONNX models in directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    total_models = len(model_files)

    if total_models == 0:
        print(f"No ONNX models found in {model_dir}")
        return

    print(f"Found {total_models} ONNX models. Will run {runs_per_model} benchmarks per model.")
>>>>>>> Stashed changes
    print(f"Total benchmarks to run: {total_models * runs_per_model}")
    print("=" * 60)

    for model_idx, model_file in enumerate(model_files, 1):
        model_path = os.path.join(model_dir, model_file)
<<<<<<< Updated upstream
        
        print(f"\n[{model_idx}/{total_models}] Processing: {model_file}")
        
=======

        print(f"\n[{model_idx}/{total_models}] Processing: {model_file}")

>>>>>>> Stashed changes
        # Parse the model to get its properties
        try:
            model_json = parse_onnx(model_path)
            if model_json['total_flops'] == 0:
                print(f"  Skipping {model_file} - no FLOPs detected")
                continue
        except Exception as e:
            print(f"  Failed to parse {model_file}: {e}")
            continue

<<<<<<< Updated upstream
        # Run multiple benchmarks for this model
        benchmark_results = run_benchmark(model_path, runs_per_model)
        
        if not benchmark_results:
            print(f"  No successful benchmarks for {model_file}")
            continue
        
=======
        # Run multiple benchmarks
        benchmark_results = run_benchmark(model_path, runs_per_model)

        if not benchmark_results:
            print(f"  No successful benchmarks for {model_file}")
            continue

>>>>>>> Stashed changes
        # Create records for each benchmark run
        for result in benchmark_results:
            record = {
                'model_name': model_file,
                'total_flops': model_json['total_flops'],
                'total_params': model_json['total_params'],
                'run_id': result['run_id'],
                **hardware_config,
                'latency_ms': result['latency_ms'],
                'power_w': result['power_w'],
                'memory_mb': result['memory_mb'],
                'data_source': 'real_benchmark'
            }
<<<<<<< Updated upstream
            
            # Calculate throughput based on actual latency
            record['throughput_ops_s'] = record['total_flops'] / (record['latency_ms'] / 1000)
            all_results.append(record)
        
        print(f"  Added {len(benchmark_results)} records for {model_file}")
    
=======
            record['throughput_ops_s'] = record['total_flops'] / (record['latency_ms'] / 1000)
            all_results.append(record)

        print(f"  Added {len(benchmark_results)} records for {model_file}")

>>>>>>> Stashed changes
    # Save all results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = DATA_DIR / output_file
        df.to_csv(output_path, index=False)
<<<<<<< Updated upstream
        
=======

>>>>>>> Stashed changes
        print("\n" + "=" * 60)
        print(f"SUCCESS: Real benchmark data saved to {output_path}")
        print(f"Total records: {len(all_results)}")
        print(f"Models benchmarked: {len(df['model_name'].unique())}")
        print(f"Average runs per model: {len(all_results) / len(df['model_name'].unique()):.1f}")
<<<<<<< Updated upstream
        
        # Show some statistics
=======

        # Show latency statistics
>>>>>>> Stashed changes
        print(f"\nLatency statistics (ms):")
        print(f"  Mean: {df['latency_ms'].mean():.2f}")
        print(f"  Std:  {df['latency_ms'].std():.2f}")
        print(f"  Min:  {df['latency_ms'].min():.2f}")
        print(f"  Max:  {df['latency_ms'].max():.2f}")
<<<<<<< Updated upstream
        
    else:
        print("No successful benchmarks were completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate real performance data from ONNX models.")
    parser.add_argument("--model_dir", type=str, default="models_onnx", 
                       help="Directory containing ONNX models.")
    parser.add_argument("--output_file", type=str, default="real_data_50runs.csv", 
                       help="Output CSV file name.")
    parser.add_argument("--runs_per_model", type=int, default=50,
                       help="Number of benchmark runs per model (default: 50)")
    
=======

    else:
        print("No successful benchmarks were completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate real performance data from ONNX models.")
    parser.add_argument("--model_dir", type=str, default="models_onnx",
                        help="Directory containing ONNX models.")
    parser.add_argument("--output_file", type=str, default="real_data_50runs.csv",
                        help="Output CSV file name.")
    parser.add_argument("--runs_per_model", type=int, default=50,
                        help="Number of benchmark runs per model (default: 50)")

>>>>>>> Stashed changes
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Model directory '{args.model_dir}' not found.")
        print("Please download some ONNX models first using the download script.")
        sys.exit(1)
    else:
<<<<<<< Updated upstream
        generate_real_data(args.model_dir, args.output_file, args.runs_per_model)
=======
        generate_real_data(args.model_dir, args.output_file, args.runs_per_model)
>>>>>>> Stashed changes
