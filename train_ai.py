import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import argparse

# --- Configuration ---
DATA_DIR = Path(__file__).parent / 'dataset_generation'
MODELS_DIR = Path(__file__).parent / 'trained_models'


    
    # Remove any repeated GPU detection calls here

def setup_device():
    """Setup device and print GPU info only once."""
    if torch.cuda.is_available():
        device = 'cuda'
        # Enable optimizations for better GPU performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Clear GPU cache
        torch.cuda.empty_cache()
        
    else:
        device = 'cpu'
        print("WARNING: CUDA not available. Using CPU training (will be much slower)")
        
    return device

# Call setup_device() only once at module level


DEVICE = setup_device()

# ==============================================================================
#  ENHANCED PYTORCH MoE MODEL DEFINITION
# ==============================================================================

class Expert(nn.Module):
    """Enhanced expert network with dropout, batch normalization, and positive outputs."""
    def __init__(self, input_size, output_size=1, dropout_rate=0.2, positive_output=True):
        super(Expert, self).__init__()
        self.positive_output = positive_output
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        # Add ReLU for positive-only outputs (latency, power, memory)
        if positive_output:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.net(x)
        return self.activation(x)


class MoEModel(nn.Module):
    """Enhanced Mixture of Experts model with better architecture."""
    def __init__(self, input_size, num_experts, targets):
        super(MoEModel, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.targets = targets
        
        # Enhanced gating network
        self.gating = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts * len(targets)),
            nn.Softmax(dim=1)
        )
        
        # Create experts for each target with appropriate constraints
        self.experts = nn.ModuleDict()
        for target in targets:
            # Use positive output constraint for physical quantities
            positive_output = target in ['latency_ms', 'power_w', 'memory_mb']
            self.experts[target] = nn.ModuleList([
                Expert(input_size, positive_output=positive_output) 
                for _ in range(num_experts)
            ])

    def forward(self, x):
        # Gating network output
        gate_outputs = self.gating(x).view(x.size(0), len(self.targets), self.num_experts)
        
        final_outputs = {}
        for i, target in enumerate(self.targets):
            target_gate = gate_outputs[:, i, :]
            
            # Get predictions from all experts
            expert_outputs = torch.stack([expert(x) for expert in self.experts[target]], dim=2)
            
            # Weight expert outputs
            weighted_output = torch.bmm(expert_outputs.squeeze(1).unsqueeze(1), target_gate.unsqueeze(2)).squeeze()
            final_outputs[target] = weighted_output
            
        return final_outputs

# ==============================================================================
#  ENHANCED TRAINING LOGIC
# ==============================================================================

def train_moe_model(data_path: Path, num_experts: int = 4, epochs: int = 150):
    """Enhanced training with all available features and better metrics."""
    print(f"Loading dataset and preparing for PyTorch training on {DEVICE}...")
    
    # GPU memory monitoring
    if DEVICE == 'cuda':
        print(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    df = pd.read_csv(data_path).dropna()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

    # Enhanced feature set including all engineered features
    base_features = [
        'total_flops', 'total_params', 'array_size', 'precision', 
        'batch_size', 'clock_ghz'
    ]
    
    engineered_features = [
        'flops_per_param', 'compute_intensity', 'array_utilization',
        'power_efficiency', 'memory_efficiency'
    ]
    
    # Use all available features
    available_features = []
    for feature in base_features + engineered_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            print(f"Warning: Feature '{feature}' not found in dataset")
    
    print(f"Using {len(available_features)} features: {available_features}")
    
    targets = ['latency_ms', 'power_w', 'memory_mb', 'throughput_ops_s']
    
    # Verify target columns exist
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"Error: Missing target columns: {missing_targets}")
        return
    
    X = df[available_features].values
    y = df[targets].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale targets too for better convergence!
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    print(f"Target scaling statistics:")
    print(f"  Original y_train range: {y_train.min(axis=0)} to {y_train.max(axis=0)}")
    print(f"  Scaled y_train range: {y_train_scaled.min(axis=0)} to {y_train_scaled.max(axis=0)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    print(f"Tensors created. Shape: X={X_train_tensor.shape}, y={y_train_tensor.shape}")

    # Data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # Initialize model
    input_size = len(available_features)
    model = MoEModel(input_size, num_experts, targets).to(DEVICE)
    
    if DEVICE == 'cuda':
        print(f"Model moved to GPU. Current memory usage: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    print(f"\n--- Training Enhanced MoE Model on {DEVICE} ---")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training for {epochs} epochs...")
    
    # Track metrics
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate weighted loss
            loss = 0
            for i, target in enumerate(targets):
                target_loss = criterion(outputs[target], batch_y[:, i])
                # Weight targets differently
                weight = 2.0 if target == 'latency_ms' else 1.0
                loss += weight * target_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                
                outputs = model(batch_X)
                loss = 0
                for i, target in enumerate(targets):
                    target_loss = criterion(outputs[target], batch_y[:, i])
                    weight = 2.0 if target == 'latency_ms' else 1.0
                    loss += weight * target_loss
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print epoch results
        if (epoch + 1) % 1 == 0:  # Print every epoch
            print(f'Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test_tensor.to(DEVICE)
        test_outputs = model(X_test_gpu)
        
        # Collect all predictions for unscaling
        test_predictions = np.zeros((len(X_test_tensor), len(targets)))
        for i, target in enumerate(targets):
            test_predictions[:, i] = test_outputs[target].cpu().numpy().flatten()
        
        # Unscale all predictions
        test_pred_unscaled = target_scaler.inverse_transform(test_predictions)
        
        print(f"\n--- Final Model Performance ---")
        for i, target in enumerate(targets):
            test_r2 = r2_score(y_test[:, i], test_pred_unscaled[:, i])
            test_mae = mean_absolute_error(y_test[:, i], test_pred_unscaled[:, i])
            
            print(f"{target}:")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Test MAE: {test_mae:.3f}")
            print(f"  Sample predictions: {test_pred_unscaled[:3, i]}")
            print(f"  Sample actual: {y_test[:3, i]}")
    
    # Save model and metadata
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / 'enhanced_moe_model.pth'
    feature_scaler_path = MODELS_DIR / 'feature_scaler.pkl'
    target_scaler_path = MODELS_DIR / 'target_scaler.pkl'
    metadata_path = MODELS_DIR / 'model_metadata.pkl'
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    
    # Save metadata for inference
    metadata = {
        'features': available_features,
        'targets': targets,
        'input_size': input_size,
        'num_experts': num_experts,
        'model_architecture': 'Enhanced MoE with Target Scaling',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'target_scaling': True
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"\n--- Training Complete ---")
    print(f"Enhanced model saved to: {model_path}")
    print(f"Feature scaler saved to: {feature_scaler_path}")
    print(f"Target scaler saved to: {target_scaler_path}")
    print(f"Model metadata saved to: {metadata_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train enhanced MoE performance prediction model.")
    parser.add_argument("--dataset", type=str, default="data/merged_training_data.csv", 
                       help="Dataset CSV file path")
    parser.add_argument("--epochs", type=int, default=150,  # Increased default
                       help="Number of training epochs")
    parser.add_argument("--experts", type=int, default=4,   # Reasonable default
                       help="Number of experts in MoE")
    
    args = parser.parse_args()
    
    dataset_path = DATA_DIR / args.dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Please make sure the merged dataset exists")
    else:
        train_moe_model(dataset_path, num_experts=args.experts, epochs=args.epochs)