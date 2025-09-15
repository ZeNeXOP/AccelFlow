import torch
import numpy as np
import joblib
from ts.torch_handler.base_handler import BaseHandler

class AccelForgeMoeHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.metadata = None
        self.targets = ['latency_ms', 'power_w', 'memory_mb', 'throughput_ops_s']

    def initialize(self, ctx):
        """Load model, scalers, and metadata"""
        self.manifest = ctx.manifest
        model_dir = ctx.system_properties.get("model_dir")
        
        print(f"📦 Loading model from: {model_dir}")
        
        try:
            # Load MoE model
            model_path = f"{model_dir}/enhanced_moe_model.pth"
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.eval()
            print("✅ MoE model loaded")

            # Load feature scaler
            self.feature_scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
            print("✅ Feature scaler loaded")

            # Load target scaler and metadata
            self.target_scaler = joblib.load(f"{model_dir}/target_scaler.pkl")
            self.metadata = joblib.load(f"{model_dir}/model_metadata.pkl")
            print("✅ Target scaler and metadata loaded")

            print(f"📊 Model metadata: {self.metadata}")

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise

    def preprocess(self, data):
        """Convert input data to feature array"""
        # Extract input from request
        inputs = data[0].get("body") or data[0].get("data")
        
        if isinstance(inputs, dict):
            # Handle JSON input: {'total_flops': 2.5e9, 'total_params': 3.5e6, ...}
            features = [
                inputs.get('total_flops', 0),
                inputs.get('total_params', 0),
                inputs.get('flops_per_param', 0),
                inputs.get('compute_intensity', 0),
                inputs.get('array_size', 8),
                0 if inputs.get('precision', 'FP16') == 'INT8' else 
                1 if inputs.get('precision', 'FP16') == 'FP16' else 2,
                inputs.get('batch_size', 1),
                inputs.get('clock_ghz', 1.0),
                inputs.get('array_utilization', 0.5),
                inputs.get('power_efficiency', 1.0),
                inputs.get('memory_efficiency', 1.0)
            ]
        else:
            # Assume already formatted feature array
            features = inputs
        
        # Convert to numpy and scale
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features_array)
        
        return torch.tensor(features_scaled, dtype=torch.float32)

    def inference(self, data):
        """Run model inference"""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def postprocess(self, inference_output):
        """Convert model output to response format"""
        results = {}
        
        # Handle each target prediction
        for i, target in enumerate(self.targets):
            pred = inference_output[target].numpy()
            
            # Inverse transform if target scaling was used during training
            if self.metadata.get('target_scaling', False):
                pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1))
            
            results[target] = pred.flatten().tolist()[0]  # Single prediction
        
        return [results]

    def handle(self, data, context):
        """Main handling method"""
        try:
            # Preprocess
            model_input = self.preprocess(data)
            
            # Inference
            model_output = self.inference(model_input)
            
            # Postprocess
            return self.postprocess(model_output)
            
        except Exception as e:
            print(f"❌ Handler error: {e}")
            raise

# For local testing
if __name__ == "__main__":
    # Test the handler
    class MockContext:
        manifest = {}
        system_properties = {"model_dir": "trained_models"}
    
    handler = AccelForgeMoeHandler()
    handler.initialize(MockContext())
    
    # Test prediction
    test_data = [{
        "body": {
            'total_flops': 2.5e9,
            'total_params': 3.5e6, 
            'flops_per_param': 714.29,
            'compute_intensity': 15.2,
            'array_size': 16,
            'precision': 'INT8',
            'batch_size': 1,
            'clock_ghz': 1.2,
            'array_utilization': 0.8,
            'power_efficiency': 1.0,
            'memory_efficiency': 1.0
        }
    }]
    
    result = handler.handle(test_data, None)
    print("🧪 Test prediction:", result)