#!/usr/bin/env python3
"""Quick check if model training was successful"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'trained_models'

required_files = [
    'enhanced_moe_model.pth',
    'feature_scaler.pkl', 
    'target_scaler.pkl',
    'model_metadata.pkl'
]

print("🔍 Checking trained model files...")
missing_files = []

for file in required_files:
    if (MODELS_DIR / file).exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (MISSING)")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Model training incomplete. Missing {len(missing_files)} files.")
    print("Run: python train_ai.py --epochs 100 --experts 4")
else:
    print(f"\n✅ Model training successful! Ready for AI-powered search.")