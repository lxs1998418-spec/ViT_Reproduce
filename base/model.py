"""
Vision Transformer (ViT) Implementation
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Pretrained weights cannot be loaded.")


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer Model"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Dropout for embeddings
        self.pos_dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.normal_(self.patch_embed.proj.bias, std=1e-6)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.norm1.weight, std=0.02)
            nn.init.normal_(block.norm2.weight, std=0.02)
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.normal_(block.attn.qkv.bias, std=1e-6)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.normal_(block.attn.proj.bias, std=1e-6)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.normal_(block.mlp.fc1.bias, std=1e-6)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.normal_(block.mlp.fc2.bias, std=1e-6)
        
        # Initialize classification head
        nn.init.normal_(self.norm.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use class token)
        cls_token_final = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_token_final)  # [B, num_classes]
        
        return logits


def vit_base_patch16_224(num_classes=1000, pretrained=False):
    """ViT-Base/16 model
    
    Args:
        num_classes: Number of output classes
        pretrained: If True, load pretrained weights from timm
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        emb_dropout=0.1
    )
    
    if pretrained:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required to load pretrained weights. Install it with: pip install timm")
        load_pretrained_weights(model, 'vit_base_patch16_224', num_classes)
    
    return model


def vit_large_patch16_224(num_classes=1000, pretrained=False):
    """ViT-Large/16 model
    
    Args:
        num_classes: Number of output classes
        pretrained: If True, load pretrained weights from timm
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
        emb_dropout=0.1
    )
    
    if pretrained:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required to load pretrained weights. Install it with: pip install timm")
        load_pretrained_weights(model, 'vit_large_patch16_224', num_classes)
    
    return model


def vit_small_patch16_224(num_classes=1000, pretrained=False):
    """ViT-Small/16 model
    
    Args:
        num_classes: Number of output classes
        pretrained: If True, load pretrained weights from timm
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        emb_dropout=0.1
    )
    
    if pretrained:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required to load pretrained weights. Install it with: pip install timm")
        load_pretrained_weights(model, 'vit_small_patch16_224', num_classes)
    
    return model


def load_pretrained_weights(model, timm_model_name, num_classes):
    """Load pretrained weights from timm library
    
    Args:
        model: Our VisionTransformer model
        timm_model_name: Name of the model in timm (e.g., 'vit_base_patch16_224')
        num_classes: Number of classes for the final classification head
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required. Install it with: pip install timm")
    
    print(f"Loading pretrained weights from timm: {timm_model_name}")
    
    # Load pretrained model from timm
    pretrained_model = timm.create_model(timm_model_name, pretrained=True, num_classes=1000)
    
    # Get state dicts
    our_state_dict = model.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    
    # Try to load matching weights
    loaded_keys = []
    missing_keys = []
    shape_mismatch_keys = []
    
    for key in our_state_dict.keys():
        if key in pretrained_state_dict:
            if our_state_dict[key].shape == pretrained_state_dict[key].shape:
                our_state_dict[key] = pretrained_state_dict[key]
                loaded_keys.append(key)
            else:
                shape_mismatch_keys.append(key)
        else:
            missing_keys.append(key)
    
    # Handle classification head separately (may have different num_classes)
    if 'head.weight' in pretrained_state_dict and 'head.bias' in pretrained_state_dict:
        pretrained_head_weight = pretrained_state_dict['head.weight']
        pretrained_head_bias = pretrained_state_dict['head.bias']
        
        if num_classes == 1000:
            # Same number of classes, should already be loaded
            if 'head.weight' not in loaded_keys:
                our_state_dict['head.weight'] = pretrained_head_weight
                our_state_dict['head.bias'] = pretrained_head_bias
                loaded_keys.extend(['head.weight', 'head.bias'])
        else:
            # Different number of classes, initialize appropriately
            if num_classes <= pretrained_head_weight.shape[0]:
                # Take first num_classes from pretrained head
                our_state_dict['head.weight'] = pretrained_head_weight[:num_classes]
                our_state_dict['head.bias'] = pretrained_head_bias[:num_classes]
                if 'head.weight' in loaded_keys:
                    loaded_keys.remove('head.weight')
                if 'head.bias' in loaded_keys:
                    loaded_keys.remove('head.bias')
                loaded_keys.extend(['head.weight (partial)', 'head.bias (partial)'])
            else:
                # Initialize new head with pretrained weights (copy first 1000 classes)
                our_state_dict['head.weight'][:1000] = pretrained_head_weight
                our_state_dict['head.bias'][:1000] = pretrained_head_bias
                if 'head.weight' in loaded_keys:
                    loaded_keys.remove('head.weight')
                if 'head.bias' in loaded_keys:
                    loaded_keys.remove('head.bias')
                loaded_keys.extend(['head.weight (extended)', 'head.bias (extended)'])
    
    # Load the state dict
    model.load_state_dict(our_state_dict, strict=False)
    
    print(f"Successfully loaded {len(loaded_keys)} pretrained layers")
    if shape_mismatch_keys:
        print(f"Warning: {len(shape_mismatch_keys)} layers have shape mismatches (not loaded)")
    if missing_keys:
        print(f"Note: {len(missing_keys)} layers not found in pretrained model (using random initialization)")
    print(f"Classification head initialized for {num_classes} classes")
    
    return model

