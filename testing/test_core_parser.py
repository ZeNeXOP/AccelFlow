import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import pytest
import onnx
from onnx import helper, TensorProto, shape_inference
from core_parser import parse_onnx  # Adjust import if needed


def create_simple_onnx_model():
    # Create a simple graph: Conv -> ReLU
    # Input: 1x3x32x32 (batch, channels, height, width)
    # Conv: kernel 3x3, out channels 8, no bias for simplicity
    # Expected params: weights = 8 * 3 * 3 * 3 = 216
    # Expected FLOPs: 2 * 3 * 3 * 3 * 8 * 30 * 30 = 2 * 3*3*3*8*900 = 2*27*8*900 = 388800
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8, 30, 30])

    weight_tensor = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[8, 3, 3, 3],
        vals=[0.0] * (8 * 3 * 3 * 3)  # Dummy values
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

def test_parse_onnx_conv_layer(temp_onnx_file):
    result = parse_onnx(temp_onnx_file)
    assert result['model_name'] == os.path.basename(temp_onnx_file).replace('.onnx', '')
    assert len(result['layers']) == 2
    conv_layer = result['layers'][0]
    assert conv_layer['type'] == 'Conv'
    assert conv_layer['params'] == 216  # Weights only, no bias
    assert conv_layer['flops'] == 388800
    relu_layer = result['layers'][1]
    assert relu_layer['type'] == 'Relu'
    assert relu_layer['params'] == 0
    assert relu_layer['flops'] == 0
    assert result['total_params'] == 216
    assert result['total_flops'] == 388800

def test_parse_onnx_no_crash_on_empty():
    # Test with an empty model (minimal graph with no nodes)
    empty_model = helper.make_model(helper.make_graph([], 'empty', [], []))
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx.save(empty_model, tmp_file.name)
    try:
        result = parse_onnx(tmp_file.name)
        assert result['layers'] == []
        assert result['total_params'] == 0
        assert result['total_flops'] == 0
    finally:
        os.unlink(tmp_file.name)