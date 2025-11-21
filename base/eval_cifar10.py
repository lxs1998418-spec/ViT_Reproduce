"""
CIFAR-10 Evaluation script for Vision Transformer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm
from model import vit_base_patch16_224, vit_small_patch16_224, vit_large_patch16_224


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_model(model_name, num_classes=10, pretrained=False):
    """Get model by name
    
    Args:
        model_name: Name of the model ('vit_small', 'vit_base', 'vit_large')
        num_classes: Number of output classes (10 for CIFAR-10)
        pretrained: If True, load pretrained weights
    """
    if model_name == 'vit_base':
        return vit_base_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit_small':
        return vit_small_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit_large':
        return vit_large_patch16_224(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_cifar10_loader(batch_size=32, num_workers=4, img_size=224, train=False):
    """Get CIFAR-10 data loader
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Target image size (ViT expects 224x224)
        train: If True, load training set; otherwise load test set
    """
    # Transform: CIFAR-10 images are 32x32, need to resize to 224x224 for ViT
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize from 32x32 to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def evaluate_cifar10(model, data_loader, device):
    """Evaluate the model on CIFAR-10
    
    Returns:
        overall_acc: Overall accuracy
        class_acc: Dictionary mapping class index to accuracy
        confusion_matrix: Confusion matrix
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    confusion_matrix = torch.zeros(10, 10)
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Per-class statistics
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    overall_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_acc = {}
    for cls in range(10):
        if class_total[cls] > 0:
            class_acc[cls] = 100. * class_correct[cls] / class_total[cls]
        else:
            class_acc[cls] = 0.0
    
    return overall_acc, class_acc, confusion_matrix


def print_results(overall_acc, class_acc, confusion_matrix):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("CIFAR-10 Evaluation Results")
    print("="*60)
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")
    
    print("\nPer-class Accuracy:")
    print("-" * 60)
    for cls in range(10):
        class_name = CIFAR10_CLASSES[cls]
        acc = class_acc[cls]
        print(f"  {class_name:15s}: {acc:6.2f}%")
    
    # Calculate average per-class accuracy
    avg_class_acc = sum(class_acc.values()) / len(class_acc)
    print("-" * 60)
    print(f"  {'Average':15s}: {avg_class_acc:6.2f}%")
    
    # Print confusion matrix (optional, can be verbose)
    print("\nConfusion Matrix (rows: true, cols: predicted):")
    print("-" * 60)
    print("      ", end="")
    for i in range(10):
        print(f"{CIFAR10_CLASSES[i][:4]:>6}", end="")
    print()
    for i in range(10):
        print(f"{CIFAR10_CLASSES[i][:6]:>6}", end="")
        for j in range(10):
            print(f"{int(confusion_matrix[i, j]):>6}", end="")
        print()
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer on CIFAR-10')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional if using --pretrained)')
    parser.add_argument('--model', type=str, default='vit_base',
                        choices=['vit_small', 'vit_base', 'vit_large'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights from timm (ignores --checkpoint)')
    parser.add_argument('--train_set', action='store_true',
                        help='Evaluate on training set instead of test set')
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.pretrained:
        parser.error("Either --checkpoint or --pretrained must be specified")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loader
    print('Loading CIFAR-10 dataset...')
    data_loader = get_cifar10_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        train=args.train_set
    )
    dataset_name = "training" if args.train_set else "test"
    print(f'Evaluating on CIFAR-10 {dataset_name} set ({len(data_loader.dataset)} images)')
    
    # Model
    print(f'\nCreating model: {args.model}')
    if args.pretrained:
        print('Loading pretrained weights from timm...')
        print('Note: Classification head will be adapted for 10 classes')
        model = get_model(args.model, num_classes=10, pretrained=True)
    else:
        model = get_model(args.model, num_classes=10, pretrained=False)
        # Load checkpoint
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Evaluate
    print('\nEvaluating...')
    overall_acc, class_acc, confusion_matrix = evaluate_cifar10(model, data_loader, device)
    
    # Print results
    print_results(overall_acc, class_acc, confusion_matrix)


if __name__ == '__main__':
    main()


