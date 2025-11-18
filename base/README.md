# Vision Transformer (ViT) Reproduction

A PyTorch implementation of Vision Transformer (ViT) from scratch, based on the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

This implementation does **not** use pre-trained weights and trains the model from scratch.

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

1. Clone this repository:
```bash
git clone <repository-url>
cd ViT_Reproduce
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

Train a ViT model from scratch:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --model vit_base \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
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

### Evaluation

Evaluate a trained model:

```bash
python eval.py \
    --checkpoint ./checkpoints/best.pth \
    --data_dir /path/to/validation/data \
    --model vit_base \
    --batch_size 32
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--data_dir`: Path to evaluation dataset directory
- `--model`: Model architecture (must match the checkpoint)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--img_size`: Image size (default: 224)

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

## Example

Train ViT-Base on ImageNet-style dataset:

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

## Notes

- This implementation trains from scratch without pre-trained weights
- For best results, use large datasets (e.g., ImageNet)
- Training ViT from scratch typically requires many epochs and large datasets
- Consider using mixed precision training for faster training on modern GPUs

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

