import torch
import numpy as np
import joblib
from pathlib import Path
import os

class AccelForgePredictor:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load trained MoE model and scalers"""
        try:
            # Load from environment variable or default path
            model_dir = os.getenv('MODEL_DIR', '/models')
            
            self.model = torch.load(
                f'{model_dir}/enhanced_moe_model.pth', 
                map_location=self.device
            )
            self.feature_scaler = joblib.load(f'{model_dir}/feature_scaler.pkl')
            self.target_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
            self.metadata = joblib.load(f'{model_dir}/model_metadata.pkl')
            
            self.model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def predict(self, instances):
        """Main prediction function for Vertex AI"""
        try:
            # Convert to numpy array
            features = np.array(instances, dtype=np.float32)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(features_tensor)
            
            # Convert to numpy and inverse scale
            results = []
            for i, target in enumerate(self.metadata['targets']):
                pred = predictions[target].cpu().numpy()
                # Inverse transform if target scaling was used
                if self.metadata.get('target_scaling', False):
                    pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1))
                results.append(pred.flatten().tolist())
            
            # Format as {latency: [], power: [], ...}
            return {
                self.metadata['targets'][i]: results[i] 
                for i in range(len(self.metadata['targets']))
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise

# Global predictor instance
predictor = AccelForgePredictor()

def predict(request):
    """Vertex AI prediction endpoint"""
    try:
        instances = request.get('instances', [])
        if not instances:
            return {'error': 'No instances provided'}
        
        results = predictor.predict(instances)
        return {'predictions': results}
        
    except Exception as e:
        return {'error': str(e)}