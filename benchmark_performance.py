from core_predictor import predict_performance_ai
import time

def benchmark_ai_predictor():
    """Benchmark AI prediction speed"""
    test_cases = 100
    print(f"⏱️  Benchmarking AI predictor with {test_cases} iterations...")
    
    # Simple test model
    model_json = {
        'total_flops': 1e9, 'total_params': 1e6,
        'flops_per_param': 1000, 'compute_intensity': 10
    }
    config = {'array_size': 16, 'precision': 'INT8', 'clock_ghz': 1.0, 'batch_size': 1}
    
    times = []
    for i in range(test_cases):
        start = time.time()
        predict_performance_ai(model_json, config)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times) * 1000  # Convert to ms
    print(f"✅ Average prediction time: {avg_time:.2f}ms")
    print(f"✅ Predictions per second: {1000/avg_time:.0f}")