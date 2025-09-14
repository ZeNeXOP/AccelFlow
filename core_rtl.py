import os
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent

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


def generate_rtl(hardware_config: dict) -> str:
    """Generate RTL code for the systolic array based on hardware configuration."""
    # Load template
    template_path = PROJECT_ROOT / "rtl_templates" / "systolic_array_template.v"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Systolic array template not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Extract configuration parameters
    array_size = hardware_config.get('array_size', 8)
    precision = hardware_config.get('precision', 'INT8')
    clock_ghz = hardware_config.get('clock_ghz', 1.0)
    
    # Determine data width based on precision
    if precision == 'INT8':
        data_width = 8
    elif precision == 'FP16':
        data_width = 16
    else:  # FP32
        data_width = 32
    
    # Replace placeholders
    rtl_code = template.replace('{{ARRAY_SIZE}}', str(array_size))
    rtl_code = rtl_code.replace('{{DATA_WIDTH}}', str(data_width))
    rtl_code = rtl_code.replace('{{CLOCK_FREQUENCY}}', f"{clock_ghz:.1f}")
    
    return rtl_code

# Example usage
if __name__ == '__main__':
    mock_config = {'array_size': 8, 'precision': 'FP16', 'model_name': 'TestModel'}
    generated = generate_rtl(mock_config)
    print(generated)
    with open('generated_rtl.v', 'w') as f:
        f.write(generated)