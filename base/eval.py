"""
Evaluation script for Vision Transformer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm
from model import vit_base_patch16_224, vit_small_patch16_224, vit_large_patch16_224


def get_model(model_name, num_classes):
    """Get model by name"""
    if model_name == 'vit_base':
        return vit_base_patch16_224(num_classes=num_classes)
    elif model_name == 'vit_small':
        return vit_small_patch16_224(num_classes=num_classes)
    elif model_name == 'vit_large':
        return vit_large_patch16_224(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_data_loader(data_dir, batch_size=32, num_workers=4, img_size=224):
    """Get data loader for evaluation"""
    
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, len(dataset.classes)


def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    overall_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_acc = {}
    for cls in class_total:
        class_acc[cls] = 100. * class_correct[cls] / class_total[cls]
    
    return overall_acc, class_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to evaluation dataset directory')
    parser.add_argument('--model', type=str, default='vit_base',
                        choices=['vit_small', 'vit_base', 'vit_large'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loader
    print('Loading data...')
    data_loader, num_classes = get_data_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.img_size
    )
    print(f'Number of classes: {num_classes}')
    
    # Model
    print(f'Creating model: {args.model}')
    model = get_model(args.model, num_classes)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print('Evaluating...')
    overall_acc, class_acc = evaluate(model, data_loader, device)
    
    print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    print(f'\nPer-class Accuracy:')
    for cls, acc in sorted(class_acc.items()):
        print(f'  Class {cls}: {acc:.2f}%')


if __name__ == '__main__':
    main()

