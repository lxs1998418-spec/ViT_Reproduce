"""
Simple test script to verify the ViT model works correctly
"""

import torch
from model import vit_base_patch16_224, vit_small_patch16_224, vit_large_patch16_224
from utils import count_parameters


def test_model(model, model_name, num_classes=1000):
    """Test a model with dummy input"""
    print(f"\nTesting {model_name}...")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {num_classes})")
    
    assert output.shape == (batch_size, num_classes), f"Output shape mismatch!"
    print(f"  ✓ {model_name} works correctly!")


if __name__ == '__main__':
    print("Testing Vision Transformer Models")
    print("=" * 50)
    
    # Test ViT-Small
    vit_small = vit_small_patch16_224(num_classes=1000)
    test_model(vit_small, "ViT-Small/16")
    
    # Test ViT-Base
    vit_base = vit_base_patch16_224(num_classes=1000)
    test_model(vit_base, "ViT-Base/16")
    
    # Test ViT-Large
    vit_large = vit_large_patch16_224(num_classes=1000)
    test_model(vit_large, "ViT-Large/16")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")

