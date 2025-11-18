"""
Utility functions for Vision Transformer
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_attention(model, image_path, device, patch_size=16, head_fusion='mean'):
    """
    Visualize attention maps from the Vision Transformer
    
    Args:
        model: Trained ViT model
        image_path: Path to input image
        device: torch device
        patch_size: Patch size used in the model
        head_fusion: How to fuse attention heads ('mean', 'max', 'min')
    """
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0).to(device)
    
    # Get attention weights
    with torch.no_grad():
        # Forward pass through patch embedding
        x = model.patch_embed(img_tensor)
        B, N, C = x.shape
        
        # Add class token
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + model.pos_embed
        x = model.pos_dropout(x)
        
        # Collect attention weights from all layers
        attentions = []
        for block in model.blocks:
            x_norm = block.norm1(x)
            qkv = block.attn.qkv(x_norm)
            B, N, _ = qkv.shape
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (block.attn.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            attentions.append(attn)
            
            x = x + block.attn(block.norm1(x))
            x = x + block.mlp(block.norm2(x))
    
    # Average attention across layers
    attn = torch.stack(attentions).mean(dim=0)  # [num_layers, batch, heads, N, N]
    
    # Extract attention to class token (first token)
    attn_to_cls = attn[:, :, :, 0, 1:].mean(dim=1)  # [num_layers, batch, N-1]
    
    # Fuse attention heads
    if head_fusion == 'mean':
        attn_to_cls = attn_to_cls.mean(dim=1)  # [num_layers, N-1]
    elif head_fusion == 'max':
        attn_to_cls = attn_to_cls.max(dim=1)[0]
    elif head_fusion == 'min':
        attn_to_cls = attn_to_cls.min(dim=1)[0]
    
    # Average across layers
    attn_to_cls = attn_to_cls.mean(dim=0)  # [N-1]
    
    # Reshape to image dimensions
    num_patches_per_side = int(np.sqrt(attn_to_cls.shape[0]))
    attn_map = attn_to_cls.reshape(num_patches_per_side, num_patches_per_side)
    attn_map = attn_map.cpu().numpy()
    
    # Resize to original image size
    from scipy.ndimage import zoom
    attn_map = zoom(attn_map, zoom=(224/num_patches_per_side, 224/num_patches_per_side), order=1)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(attn_map, cmap='hot', interpolation='nearest')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    return attn_map


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f'Checkpoint saved to {filename}')


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


if __name__ == '__main__':
    from model import vit_base_patch16_224
    
    # Example: count parameters
    model = vit_base_patch16_224()
    num_params = count_parameters(model)
    print(f'Number of parameters: {num_params:,}')

