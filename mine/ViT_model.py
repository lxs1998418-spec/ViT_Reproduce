import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # create patch embeddings by conv2d
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: [Batch_size, Channel, Height, Width] -> [Batch_size, embed_dim, H', W']
        x = self.proj(x) 
        Batch_size, Channel, Height, Width = x.shape
        # [Batch_size, embed_dim, H', W'] -> [Batch_size, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)  
        return x



class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    # embed_dim must be divisible by num_heads
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [Batch_size, 197, 768]
        B, N, C = x.shape
        
        # Generate Q, K, V
        # [Batch_size, 197, 768] -> [Batch_size, 197, 3, 12, 64] -> [3, Batch_size, 12, 197, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        # [3, Batch_size, 12, 197, 64] -> [Batch_size, 12, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # [Batch_size, 12, 197, 64] -> [Batch_size, 12, 197, 197]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # transfer attention to values
        # [Batch_size, 12, 197, 197] -> [Batch_size, 197, 768]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    """
    Feed-forward network
    embed_dim -> hidden_dim(expand mlp_ratio times) -> embed_dim
    """
    
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
    """Transformer Block Only Encoder"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # x shape: [Batch_size, num_patches+1, embed_dim]
        # residual connection
        x = x + self.attn(self.norm1(x))
        # residual connection
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
        
        # Patch embedding [Batch_size, num_patches, embed_dim]
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding 
        # num_patches + 1  including the class token
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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        Batch_size = x.shape[0]
        
        # Patch embedding
        # [Batch_size, Channel, Height, Width] -> [Batch_size, num_patches, embed_dim]
        x = self.patch_embed(x)  

        # Add class token
        cls_tokens = self.cls_token.expand(Batch_size, -1, -1)  # [Batch_size, 1, embed_dim]
        
        # [Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches+1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        # [Batch_size, num_patches+1, embed_dim] -> [Batch_size, num_patches+1, embed_dim]
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # use class tokens for classification
        # [Batch_size, num_patches+1, embed_dim] -> [Batch_size, embed_dim]
        cls_token_final = x[:, 0]  

        # [Batch_size, embed_dim] -> [Batch_size, num_classes]
        class_vectors = self.head(cls_token_final)  
        
        return class_vectors