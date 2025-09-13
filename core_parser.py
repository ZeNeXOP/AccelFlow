import onnx
import os
import numpy as np


def parse_onnx(onnx_model_path):
    """
    Parses an ONNX model file to extract its architecture, total FLOPs, and parameters.
    It specifically calculates metrics for Conv, MatMul, and Gemm layers.

    Args:
        onnx_model_path (str): The file path to the .onnx model.

    Returns:
        dict: A dictionary containing the model's name, total FLOPs, total parameters,
              and a list of its layers with their individual metrics.
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Model file not found at: {onnx_model_path}")
    try:
        model = onnx.load(onnx_model_path)
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Could not load or infer shape for model {onnx_model_path}. Error: {e}")
        return {
            "model_name": os.path.basename(onnx_model_path).replace('.onnx', ''),
            "total_flops": 0,
            "total_params": 0,
            "layers": []
        }

    # Collect all tensor shapes from the graph
    shapes = {}
    for initializer in inferred_model.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    for vi in inferred_model.graph.value_info:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for input_val in inferred_model.graph.input:
        if input_val.name not in shapes:
            shapes[input_val.name] = [d.dim_value for d in input_val.type.tensor_type.shape.dim]
    for output_val in inferred_model.graph.output:
        if output_val.name not in shapes:
            shapes[output_val.name] = [d.dim_value for d in output_val.type.tensor_type.shape.dim]

    model_name = os.path.basename(onnx_model_path).replace('.onnx', '')

    layers = []
    total_flops = 0
    total_params = 0

    for i, node in enumerate(inferred_model.graph.node):
        layer_name = node.name if node.name else f"{node.op_type}_{i}"
        layer_type = node.op_type
        flops = 0
        params = 0

        if layer_type == 'Conv':
            input_name, weight_name, output_name = node.input[0], node.input[1], node.output[0]
            input_shape = shapes.get(input_name, [])
            weight_shape = shapes.get(weight_name, [])
            output_shape = shapes.get(output_name, [])

            # Combined check to ensure all shapes are valid before calculating
            if input_shape and output_shape and weight_shape and len(output_shape) == 4:
                attrs = {attr.name: attr for attr in node.attribute}
                kernel_shape = attrs['kernel_shape'].ints
                kx, ky = kernel_shape[0], kernel_shape[1]
                group = attrs.get('group', 1).i if 'group' in attrs else 1

                cin, cout = input_shape[1], output_shape[1]
                hout, wout = output_shape[2], output_shape[3]

                # FLOPs calculation (2 * MACs)
                flops = 2 * (cin // group) * kx * ky * cout * hout * wout

                # Params: weights + bias if present
                params = np.prod(weight_shape)
                if len(node.input) > 2:  # Check for bias
                    bias_name = node.input[2]
                    bias_shape = shapes.get(bias_name, [])
                    if bias_shape:
                        params += bias_shape[0]

        elif layer_type in ['MatMul', 'Gemm']:
            input_name, weight_name = node.input[0], node.input[1]
            input_shape = shapes.get(input_name, [])
            weight_shape = shapes.get(weight_name, [])

            if input_shape and weight_shape and len(input_shape) >= 2 and len(weight_shape) == 2:
                in_features = input_shape[-1]
                out_features = weight_shape[0]     
                # FLOPs: 2 * M * K * N (approximated for batch)
                batch_size = input_shape[0] if len(input_shape) > 1 else 1
                flops = 2 * batch_size * in_features * out_features

                # Params: weights + bias if present
                params = np.prod(weight_shape)
                if len(node.input) > 2:
                    bias_shape = shapes.get(node.input[2], [])
                    if bias_shape:
                        params += bias_shape[0]

        layers.append({
            "layer_name": layer_name,
            "type": layer_type,
            "flops": int(flops),
            "params": int(params)
        })
        total_flops += flops
        total_params += params

    return {
        "model_name": model_name,
        "total_flops": int(total_flops),
        "total_params": int(total_params),
        "layers": layers
    }

if __name__ == '__main__':
    # Example usage:
    # Create a dummy ONNX file for testing if you don't have one.
    # Make sure you have a model named 'mobilenetv2-7.onnx' in a 'models_onnx' subfolder.
    
    # Create a dummy model directory if it doesn't exist
    if not os.path.exists('models_onnx'):
        os.makedirs('models_onnx')

    # Note: You need to download a real ONNX model for this to work.
    # For example, download from: https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
    model_path = 'models_onnx/mobilenetv2-7.onnx'

    if os.path.exists(model_path):
        model_info = parse_onnx(model_path)
        import json
        print(json.dumps(model_info, indent=2))
    else:
        print(f"Test model not found at '{model_path}'.")
        print("Please download an ONNX model to test the parser.")