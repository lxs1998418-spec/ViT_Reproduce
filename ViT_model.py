import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except ImportError:
    print("Warning: 'timm' library not found. Please install it using: pip install timm")
    timm = None
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is missing
    print("Warning: 'tqdm' not found. Progress bars will be disabled.")
    def tqdm(x, desc=None): return x


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


def load_pretrained_weights(model, timm_model_name, num_classes):
    """Load pretrained weights from timm library
    
    Args:
        model: my VisionTransformer model
        timm_model_name: Name of the model in timm (e.g., 'vit_base_patch16_224')
        num_classes: Number of classes for the final classification head
    
    Note: If num_classes != 1000, the classification head will be randomly initialized.
    This means you MUST fine-tune the model on your target dataset for good performance.
    """
    print(f"Loading pretrained weights from timm: {timm_model_name}")
    
    # Load pretrained model from timm
    pretrained_model = timm.create_model(timm_model_name, pretrained=True, num_classes=1000)
    
    # Get state dicts
    our_state_dict = model.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    
    # Debug: Print key names to understand the structure
    print(f"\nDebug: Our model has {len(our_state_dict)} parameters")
    print(f"Debug: Pretrained model has {len(pretrained_state_dict)} parameters")
    print(f"Debug: Sample our keys (first 10):")
    for i, key in enumerate(list(our_state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    print(f"Debug: Sample pretrained keys (first 10):")
    for i, key in enumerate(list(pretrained_state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
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
                shape_mismatch_keys.append(f"{key} (our: {our_state_dict[key].shape} vs pretrained: {pretrained_state_dict[key].shape})")
        else:
            missing_keys.append(key)
    
    # Handle classification head separately
    if 'head.weight' in pretrained_state_dict and 'head.bias' in pretrained_state_dict:
        pretrained_head_weight = pretrained_state_dict['head.weight']
        pretrained_head_bias = pretrained_state_dict['head.bias']
        
        if num_classes == 1000:
            if 'head.weight' not in loaded_keys:
                our_state_dict['head.weight'] = pretrained_head_weight
                our_state_dict['head.bias'] = pretrained_head_bias
                loaded_keys.extend(['head.weight', 'head.bias'])
        else:
            print(f"Note: Re-initializing head for {num_classes} classes (pretrained had {pretrained_head_weight.shape[0]})")
            print(f"WARNING: Classification head is randomly initialized. Model MUST be fine-tuned for good performance!")
            
    # Load the state dict
    model.load_state_dict(our_state_dict, strict=False)
    
    print(f"\nSuccessfully loaded {len(loaded_keys)} pretrained layers")
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys not found in pretrained model (first 5): {missing_keys[:5]}")
    if shape_mismatch_keys:
        print(f"Warning: {len(shape_mismatch_keys)} layers have shape mismatches (first 3):")
        for mismatch in shape_mismatch_keys[:3]:
            print(f"  {mismatch}")
    
    # Critical check: verify key layers were loaded
    critical_keys = ['patch_embed.proj.weight', 'pos_embed', 'cls_token']
    print(f"\nCritical layer check:")
    for key in critical_keys:
        if key in loaded_keys:
            print(f"  ✓ Loaded: {key}")
        else:
            print(f"  ✗ WARNING: NOT loaded: {key}")
    
    return model


def get_dataloader(dataset_name='cifar10', batch_size=32, num_workers=2, img_size=224, train=False):
    """Get DataLoader for CIFAR-10 or CIFAR-100"""
    
    # Transform: Resize to 224x224 for ViT, ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root = './data'
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        classes = [str(i) for i in range(100)] # CIFAR-100 has 100 classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, classes


def evaluate_model(model, data_loader, device, classes):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print(f'Accuracy: {acc:.2f}%')
    return acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT on CIFAR-10/100')
    parser.add_argument('--dataset', type=str, default='all', choices=['cifar10', 'cifar100', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='timm model name')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    datasets_to_eval = []
    if args.dataset == 'all':
        datasets_to_eval = ['cifar10', 'cifar100']
    else:
        datasets_to_eval = [args.dataset]
        
    for ds_name in datasets_to_eval:
        print(f"\n{'='*20} Evaluating on {ds_name.upper()} {'='*20}")
        
        if ds_name == 'cifar10':
            num_classes = 10
        else:
            num_classes = 100
            
        # Create model
        print(f"Creating model {args.model_name} for {num_classes} classes...")
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0
        )
        
        # Load weights
        model = load_pretrained_weights(model, args.model_name, num_classes)
        model = model.to(device)
        
        # Load data
        print(f"Loading {ds_name} test set...")
        loader, classes = get_dataloader(ds_name, batch_size=args.batch_size, train=False)
        
        # Evaluate
        evaluate_model(model, loader, device, classes)

if __name__ == '__main__':
    main()
