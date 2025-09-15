import os
# Test 1: Verify AI Model Loading
print("🧪 TEST 1: AI Model Loading")
from core_predictor import load_trained_moe
model, feature_scaler, target_scaler, metadata = load_trained_moe()
print(f"✅ Model loaded: {model is not None}")
print(f"✅ Feature scaler: {feature_scaler is not None}")
print(f"✅ Target scaler: {target_scaler is not None}")
print(f"✅ Metadata: {metadata}")

# Test 2: AI Prediction Edge Cases
print("\n🧪 TEST 2: AI Prediction Edge Cases")
from core_predictor import predict_performance_ai

# Edge case: Zero FLOPs
zero_model = {'total_flops': 0, 'total_params': 0, 'flops_per_param': 0, 'compute_intensity': 0}
hw_config = {'array_size': 16, 'precision': 'INT8', 'clock_ghz': 1.0, 'batch_size': 1}
result1 = predict_performance_ai(zero_model, hw_config)
print(f"✅ Zero FLOPs handled: {result1['latency_ms'] == float('inf')}")

# Edge case: Very large model
large_model = {'total_flops': 1e15, 'total_params': 1e12, 'flops_per_param': 1000, 'compute_intensity': 50}
result2 = predict_performance_ai(large_model, hw_config)
print(f"✅ Large model handled: {result2['latency_ms'] > 0}")

# Test 3: RTL Generation
print("\n🧪 TEST 3: RTL Generation Edge Cases")
from core_rtl import generate_rtl

# Normal case
normal_config = {'array_size': 16, 'precision': 'INT8', 'clock_ghz': 1.0}
rtl1 = generate_rtl(normal_config)
print(f"✅ Normal RTL generated: {len(rtl1) > 0 and 'module' in rtl1}")

# Edge case: Invalid precision
edge_config = {'array_size': 8, 'precision': 'INVALID', 'clock_ghz': 1.0}
rtl2 = generate_rtl(edge_config)
print(f"✅ Invalid precision handled: {len(rtl2) > 0}")

# Test 4: Parser Edge Cases
print("\n🧪 TEST 4: Parser Edge Cases")
import tempfile
import onnx
from onnx import helper, TensorProto
from core_parser import parse_onnx

# Create empty model
empty_model = helper.make_model(helper.make_graph([], 'empty', [], []))
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    onnx.save(empty_model, f.name)
    result = parse_onnx(f.name)
    print(f"✅ Empty model handled: {result['total_flops'] == 0}")
    os.unlink(f.name)

# Test 5: Search Edge Cases
print("\n🧪 TEST 5: Search Edge Cases")
from core_search import run_hybrid_search

# Impossible constraints
impossible_constraints = {'max_latency_ms': 0.0001, 'max_power_w': 0.0001}
results = run_hybrid_search(large_model, impossible_constraints)
print(f"✅ Impossible constraints handled: {len(results) == 0}")

# Test 6: Full Pipeline Integration
print("\n🧪 TEST 6: Full Pipeline Integration")
from pipeline import run_full_pipeline
import os

# Test with one of your real ONNX models
onnx_path = 'dataset_generation/models_onnx/mobilenetv2-7.onnx'
if os.path.exists(onnx_path):
    constraints = {'max_latency_ms': 100, 'max_power_w': 10}
    results = run_full_pipeline(onnx_path, constraints)
    print(f"✅ Full pipeline completed: {len(results) > 0}")
    if results:
        print(f"   Best config: {results[0]['array_size']}x{results[0]['array_size']} {results[0]['precision']}")
else:
    print("⚠️  Real ONNX model not found for full test")

# Test 7: Error Handling
print("\n🧪 TEST 7: Error Handling")
try:
    # Non-existent file
    run_full_pipeline('nonexistent.onnx', {'max_latency_ms': 50})
    print("❌ Should have raised error for non-existent file")
except FileNotFoundError:
    print("✅ Proper error handling for non-existent files")

print("\n🎯 ALL INTEGRATION TESTS COMPLETED!")