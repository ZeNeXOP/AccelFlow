import numpy as np
import pandas as pd
from core_predictor import predict_performance_simple, predict_performance_ai
from core_parser import parse_onnx
import time

def run_ab_testing():
    """Compare AI predictor vs simple predictor performance"""
    print("🚀 STARTING A/B TESTING: AI vs Simple Predictor")
    print("=" * 60)
    
    # Test with real ONNX models from your dataset
    test_models = [
        'dataset_generation/models_onnx/mobilenetv2-7.onnx',
        'dataset_generation/models_onnx/resnet50-v2-7.onnx', 
        'dataset_generation/models_onnx/squeezenet1.1-7.onnx'
        'dataset_generation/models_onnx/densenet-9.onnx'
        'dataset_generation/models_onnx/googlenet-9.onnx'
    ]
    
    test_configs = [
        {'array_size': 8, 'precision': 'INT8', 'clock_ghz': 1.0, 'batch_size': 1},
        {'array_size': 16, 'precision': 'FP16', 'clock_ghz': 1.5, 'batch_size': 4},
        {'array_size': 32, 'precision': 'FP32', 'clock_ghz': 2.0, 'batch_size': 8}
    ]
    
    results = []
    
    for model_path in test_models:
        try:
            print(f"\n📊 Testing model: {model_path.split('/')[-1]}")
            model_json = parse_onnx(model_path)
            
            for config in test_configs:
                print(f"   Hardware: {config['array_size']}x{config['array_size']} {config['precision']}")
                
                # Test Simple Predictor
                start_time = time.time()
                simple_result = predict_performance_simple(model_json, config)
                simple_time = time.time() - start_time
                
                # Test AI Predictor  
                start_time = time.time()
                ai_result = predict_performance_ai(model_json, config)
                ai_time = time.time() - start_time
                
                # Calculate differences
                latency_diff = ai_result['latency_ms'] - simple_result['latency_ms']
                power_diff = ai_result['power_w'] - simple_result['power_w']
                memory_diff = ai_result['memory_mb'] - simple_result['memory_mb']
                
                results.append({
                    'model': model_path.split('/')[-1],
                    'array_size': config['array_size'],
                    'precision': config['precision'],
                    'simple_latency': simple_result['latency_ms'],
                    'ai_latency': ai_result['latency_ms'],
                    'latency_diff': latency_diff,
                    'simple_power': simple_result['power_w'],
                    'ai_power': ai_result['power_w'],
                    'power_diff': power_diff,
                    'simple_memory': simple_result['memory_mb'],
                    'ai_memory': ai_result['memory_mb'],
                    'memory_diff': memory_diff,
                    'simple_time_ms': simple_time * 1000,
                    'ai_time_ms': ai_time * 1000
                })
                
        except Exception as e:
            print(f"   ❌ Error testing {model_path}: {e}")
    
    # Analyze results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("📈 A/B TEST RESULTS SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 Total test cases: {len(df)}")
        print(f"⏱️  Average prediction time:")
        print(f"   - Simple: {df['simple_time_ms'].mean():.2f}ms")
        print(f"   - AI: {df['ai_time_ms'].mean():.2f}ms")
        
        # Accuracy comparison (assuming AI is more accurate)
        print(f"\n🎯 Average differences (AI - Simple):")
        print(f"   - Latency: {df['latency_diff'].mean():.2f}ms")
        print(f"   - Power: {df['power_diff'].mean():.2f}W")  
        print(f"   - Memory: {df['memory_diff'].mean():.2f}MB")
        
        # Save detailed results
        df.to_csv('ab_test_results.csv', index=False)
        print(f"\n💾 Detailed results saved to 'ab_test_results.csv'")
        
        return df
    else:
        print("❌ No results collected")
        return None

if __name__ == "__main__":
    run_ab_testing()