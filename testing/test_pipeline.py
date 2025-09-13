import os
import sys
import tempfile
import pytest
import onnx
from onnx import helper, TensorProto, shape_inference

# Add project root to path for imports (same as in test_core_parser.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_parser import parse_onnx
from core_predictor import predict_performance_simple
from core_search import run_simple_search
from pipeline import run_full_pipeline  # Assumes you added pipeline.py as suggested

def create_simple_onnx_model():
    # Same simple Conv + ReLU model as in test_core_parser.py
    # Expected: total_flops ~388800, total_params=216
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 224, 224])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 222, 222])

    weight_tensor = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[32, 16, 3, 3],
        vals=[0.0] * (32 * 16 * 3 * 3)
    )

    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight'],
        outputs=['conv_out'],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )

    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['output']
    )

    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv_relu',
        [input_tensor],
        [output_tensor],
        initializer=[weight_tensor]
    )

    model = helper.make_model(graph)
    inferred_model = shape_inference.infer_shapes(model)
    return inferred_model

@pytest.fixture
def temp_onnx_file():
    model = create_simple_onnx_model()
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx.save(model, tmp_file.name)
    yield tmp_file.name
    os.unlink(tmp_file.name)

def test_integration_predictor(temp_onnx_file):
    # Test Module 1 → Module 2 integration
    model_json = parse_onnx(temp_onnx_file)
    hardware_config = {'array_size': 16, 'precision': 'INT8', 'clock_ghz': 1.0, 'batch_size': 1}
    performance = predict_performance_simple(model_json, hardware_config)
    
    assert 'latency_ms' in performance
    assert performance['latency_ms'] >= 0  # Positive plausible value
    assert performance['power_w'] >= 0
    # Specific check based on simple model (flops=388800, adjust if formula changes)
    # assert abs(performance['latency_ms'] - 0.1) < 0.1  # Approximate, based on formula

def test_integration_search(temp_onnx_file):
    # Test Module 1 → Module 2 → Module 3 integration
    model_json = parse_onnx(temp_onnx_file)
    constraints = {'max_latency_ms': 1.0, 'max_power_w': 1.0}
    top_configs = run_simple_search(model_json, constraints)
    
    assert isinstance(top_configs, list)
    assert len(top_configs) <= 3  # Top 3 max
    if top_configs:
        for config in top_configs:
            assert config['latency_ms'] <= constraints['max_latency_ms']
            assert config['power_w'] <= constraints['max_power_w']

def test_full_pipeline(temp_onnx_file):
    # Test complete E2E pipeline
    constraints = {'max_latency_ms': 1.0, 'max_power_w': 1.0}
    top_configs = run_full_pipeline(temp_onnx_file, constraints)
    
    assert isinstance(top_configs, list)
    if top_configs:
        assert len(top_configs) > 0
        for config in top_configs:
            assert 'array_size' in config
            assert config['latency_ms'] >= 0 and config['latency_ms'] <= constraints['max_latency_ms']


def test_pipeline_edge_case_empty_model():
    # Test with empty model (should handle gracefully, return no configs)
    empty_model = helper.make_model(helper.make_graph([], 'empty', [], []))
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx.save(empty_model, tmp_file.name)
    try:
        constraints = {'max_latency_ms': 50, 'max_power_w': 5}
        top_configs = run_full_pipeline(tmp_file.name, constraints)
        assert len(top_configs) == 0  # No valid configs for empty model
    finally:
        os.unlink(tmp_file.name)

def test_pipeline_invalid_constraints(temp_onnx_file):
    # Test tight constraints (should return empty list)
    tight_constraints = {'max_latency_ms': 0.0001, 'max_power_w': 0.0001}
    top_configs = run_full_pipeline(temp_onnx_file, tight_constraints)
    assert len(top_configs) == 0