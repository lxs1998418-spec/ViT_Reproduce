"""
Quick start script for Google Colab
Run this cell in Colab to quickly set up and test the ViT project
"""

# Install dependencies
import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "torch", "torchvision", "numpy", "Pillow", 
                      "matplotlib", "tqdm", "scipy", "timm"])
print("✓ Dependencies installed!")

# Setup paths
import sys
import os

# Add base directory to path
current_dir = os.getcwd()
base_path = os.path.join(current_dir, 'base')
if os.path.exists(base_path):
    sys.path.append(base_path)
    print(f"✓ Added {base_path} to Python path")
else:
    print(f"⚠ Warning: {base_path} not found. Make sure you're in the project root.")

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('data', exist_ok=True)
print("✓ Directories created!")

# Test import
try:
    from model import vit_base_patch16_224
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Quick test
    model = vit_base_patch16_224(num_classes=10)
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"✓ Model test passed! Output shape: {output.shape}")
    print("\n🎉 Setup complete! You can now use the ViT models.")
    
except ImportError as e:
    print(f"⚠ Import error: {e}")
    print("Make sure the model.py file is in the correct location.")
except Exception as e:
    print(f"⚠ Error: {e}")


