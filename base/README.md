# Vision Transformer (ViT) Reproduction

A PyTorch implementation of Vision Transformer (ViT) from scratch, based on the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

This implementation supports both training from scratch and loading pretrained weights from the `timm` library.

## Features

- Complete ViT implementation with:
  - Patch embedding
  - Multi-head self-attention
  - Transformer encoder blocks
  - Classification head
- Support for ViT-Small, ViT-Base, and ViT-Large architectures
- Training script with data augmentation
- Evaluation script
- Utility functions for model analysis

## Installation

### Local Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ViT_Reproduce
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab

For running on Google Colab, see [COLAB_GUIDE.md](../COLAB_GUIDE.md) or use the provided notebook `ViT_Colab_Example.ipynb`.

Quick start in Colab:
```python
# Install dependencies
!pip install -q torch torchvision numpy Pillow matplotlib tqdm scipy timm

# Clone or upload project
!git clone https://github.com/yourusername/ViT_Reproduce.git
%cd ViT_Reproduce

# Setup paths
import sys
sys.path.append('/content/ViT_Reproduce/base')
```

## Dataset Preparation

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### Training

**Train from scratch:**
```bash
python train.py \
    --data_dir /path/to/your/data \
    --model vit_base \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

**Train with pretrained weights (recommended for faster convergence):**
```bash
python train.py \
    --data_dir /path/to/your/data \
    --model vit_base \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --pretrained \
    --save_dir ./checkpoints
```

**Arguments:**
- `--data_dir`: Path to dataset directory (should contain `train/` and `val/` folders)
- `--model`: Model architecture (`vit_small`, `vit_base`, or `vit_large`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 0.1)
- `--warmup_epochs`: Number of warmup epochs (default: 10)
- `--save_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--num_workers`: Number of data loading workers (default: 4)
- `--img_size`: Image size (default: 224)
- `--resume`: Path to checkpoint to resume training from (optional)
- `--pretrained`: Load pretrained weights from timm (default: False)

### Evaluation

**Evaluate a trained checkpoint:**
```bash
python eval.py \
    --checkpoint ./checkpoints/best.pth \
    --data_dir /path/to/validation/data \
    --model vit_base \
    --batch_size 32
```

**Evaluate using pretrained weights directly:**
```bash
python eval.py \
    --pretrained \
    --data_dir /path/to/validation/data \
    --model vit_base \
    --batch_size 32
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (optional if using `--pretrained`)
- `--data_dir`: Path to evaluation dataset directory
- `--model`: Model architecture (must match the checkpoint or pretrained model)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--img_size`: Image size (default: 224)
- `--pretrained`: Use pretrained weights from timm (ignores `--checkpoint`)

### CIFAR-10 Evaluation

**Evaluate on CIFAR-10 test set using pretrained weights:**
```bash
python eval_cifar10.py \
    --pretrained \
    --model vit_base \
    --batch_size 32
```

**Evaluate on CIFAR-10 using a trained checkpoint:**
```bash
python eval_cifar10.py \
    --checkpoint ./checkpoints/best.pth \
    --model vit_base \
    --batch_size 32
```

**Arguments for CIFAR-10 evaluation:**
- `--checkpoint`: Path to model checkpoint (optional if using `--pretrained`)
- `--model`: Model architecture (`vit_small`, `vit_base`, or `vit_large`)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--img_size`: Image size (default: 224)
- `--pretrained`: Use pretrained weights from timm (ignores `--checkpoint`)
- `--train_set`: Evaluate on training set instead of test set (default: False)

**Note:** CIFAR-10 images are 32x32, which will be automatically resized to 224x224 for ViT. The dataset will be automatically downloaded on first use.

### Model Architectures

The implementation supports three model sizes:

1. **ViT-Small/16**: 384 embedding dimension, 12 layers, 6 heads
2. **ViT-Base/16**: 768 embedding dimension, 12 layers, 12 heads
3. **ViT-Large/16**: 1024 embedding dimension, 24 layers, 16 heads

## Model Architecture

The Vision Transformer consists of:

1. **Patch Embedding**: Divides input images into fixed-size patches and projects them to embedding dimensions
2. **Positional Embedding**: Adds learnable positional embeddings
3. **Class Token**: A learnable classification token prepended to patch embeddings
4. **Transformer Encoder**: Stack of transformer blocks with:
   - Multi-head self-attention
   - Layer normalization
   - MLP (feed-forward network)
5. **Classification Head**: Linear layer for final classification

## Training Details

- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Cosine annealing with warmup
- **Data Augmentation**: Random resized crop, horizontal flip, color jitter
- **Normalization**: ImageNet mean and std normalization

## Checkpoints

During training, the following checkpoints are saved:
- `latest.pth`: Latest checkpoint after each epoch
- `best.pth`: Best checkpoint based on validation accuracy
- `history.json`: Training history (loss and accuracy)

## Examples

### Train ViT-Base on ImageNet-style dataset:

```bash
python train.py \
    --data_dir ./data \
    --model vit_base \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.001 \
    --warmup_epochs 10 \
    --save_dir ./checkpoints
```

### Evaluate on CIFAR-10 with pretrained weights:

```bash
python eval_cifar10.py \
    --pretrained \
    --model vit_base \
    --batch_size 64
```

This will automatically download CIFAR-10 dataset and evaluate the pretrained ViT-Base model on it. The script will show:
- Overall accuracy
- Per-class accuracy for all 10 CIFAR-10 classes
- Confusion matrix

## Pretrained Weights

This implementation supports loading pretrained weights from the `timm` library. The pretrained weights are automatically downloaded from Hugging Face when you use the `--pretrained` flag.

**Benefits of using pretrained weights:**
- Faster convergence during training
- Better performance, especially on small datasets
- Can be fine-tuned for your specific task

**Supported pretrained models:**
- `vit_small_patch16_224`: ViT-Small pretrained on ImageNet-1k
- `vit_base_patch16_224`: ViT-Base pretrained on ImageNet-1k
- `vit_large_patch16_224`: ViT-Large pretrained on ImageNet-1k

When using pretrained weights with a different number of classes, the classification head will be initialized appropriately:
- If your dataset has fewer classes, the head will use the first N classes from the pretrained head
- If your dataset has more classes, the head will be extended with the pretrained weights for the first 1000 classes

## Notes

- **Training from scratch**: Requires large datasets (e.g., ImageNet) and many epochs
- **Using pretrained weights**: Recommended for most use cases, especially with smaller datasets
- Consider using mixed precision training for faster training on modern GPUs
- The pretrained weights are downloaded automatically from Hugging Face via timm

## License

This implementation is provided for educational and research purposes.

## Citation

If you use this code, please cite the original ViT paper:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

