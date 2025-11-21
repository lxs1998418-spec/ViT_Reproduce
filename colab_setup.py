"""
Setup script for Google Colab
Run this in Colab to install dependencies and setup the project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "torch>=1.9.0", "torchvision>=0.10.0", 
                          "numpy>=1.21.0", "Pillow>=8.3.0", 
                          "matplotlib>=3.4.0", "tqdm>=4.62.0", 
                          "scipy>=1.7.0", "timm>=0.6.0"])
    print("✓ Requirements installed!")

def setup_colab():
    """Setup project structure in Colab"""
    print("Setting up project structure...")
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("✓ Project structure created!")
    print("\nProject is ready to use!")
    print("You can now import and use the ViT models.")

if __name__ == "__main__":
    install_requirements()
    setup_colab()


