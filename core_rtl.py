import os
from pathlib import Path

# Define the path to the templates directory, making it robust
TEMPLATE_DIR = Path(__file__).parent / 'rtl_templates'
TEMPLATE_FILE = 'systolic_array_template.v'

def generate_verilog(config: dict, model_name: str) -> str:
    """
    Generates a Verilog file from a template based on a hardware configuration.

    Args:
        config (dict): The recommended hardware configuration.
                       Example: {'array_size': 12, 'precision': 'INT8'}
        model_name (str): The name of the ML model for header comments.

    Returns:
        str: The complete, parameterized Verilog code as a string.
             Returns an error string if the template file is not found.
    """
    template_path = os.path.join(TEMPLATE_DIR, TEMPLATE_FILE)
    
    if not os.path.exists(template_path):
        return f"// ERROR: Verilog template not found at {template_path}"

    with open(template_path, 'r') as f:
        template_code = f.read()

    # Get values from the config dictionary, with safe defaults
    array_size = config.get('array_size', 8)
    precision = config.get('precision', 'INT8')
    
    # Map precision string to bit width
    precision_map = {'FP32': 32, 'FP16': 16, 'INT8': 8}
    data_width = precision_map.get(precision, 8)

    # Perform the replacements
    generated_code = template_code.replace('__DATA_WIDTH__', str(data_width))
    generated_code = generated_code.replace('__ARRAY_SIZE__', str(array_size))
    generated_code = generated_code.replace('__MODEL_NAME__', model_name)
    
    return generated_code


def generate_rtl(config: dict) -> str:
    """
    Generates Verilog RTL code by loading the systolic array template and replacing placeholders.

    Args:
        config (dict): Hardware config, e.g., {'array_size': 4, 'precision': 'INT8'}.

    Returns:
        str: Generated Verilog code.
    """
    # Map precision to bitwidth
    precision_to_bitwidth = {'INT8': 8, 'FP16': 16, 'FP32': 32}
    bitwidth = precision_to_bitwidth.get(config.get('precision', 'INT8'), 8)
    
    # Load the systolic array template
    template_path = os.path.join('rtl_templates', 'systolic_array_template.v')
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Systolic array template not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders (using __PLACEHOLDER__ style from your output)
    rtl_code = template.replace('__ARRAY_SIZE__', str(config.get('array_size', 4)))
    rtl_code = rtl_code.replace('__DATA_WIDTH__', str(bitwidth))
    rtl_code = rtl_code.replace('__ACCUM_WIDTH__', str(bitwidth * 4))
    rtl_code = rtl_code.replace('__MODEL_NAME__', config.get('model_name', 'GeneratedModel'))
    
    # Prepend includes for modularity
    includes = '`include "mac_unit_template.v"\n'
    rtl_code = includes + rtl_code
    
    return rtl_code

# Example usage
if __name__ == '__main__':
    mock_config = {'array_size': 8, 'precision': 'FP16', 'model_name': 'TestModel'}
    generated = generate_rtl(mock_config)
    print(generated)
    with open('generated_rtl.v', 'w') as f:
        f.write(generated)