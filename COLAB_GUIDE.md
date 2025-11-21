# Google Colab 使用指南

本指南将帮助您在 Google Colab 上运行 Vision Transformer 项目。

## 快速开始（最简单的方法）

### 使用提供的 Notebook

1. 打开 Google Colab
2. 上传 `ViT_Colab_Example.ipynb` 到 Colab
3. 按照 notebook 中的步骤执行

### 使用快速设置脚本

```python
# 在 Colab 中运行这个 cell
exec(open('colab_quick_start.py').read())
```

## 方法一：使用 GitHub（推荐）

### 步骤 1: 上传项目到 GitHub

1. 将项目推送到 GitHub 仓库
2. 确保所有必要的文件都在仓库中

### 步骤 2: 在 Colab 中克隆项目

```python
# 克隆仓库
!git clone https://github.com/yourusername/ViT_Reproduce.git

# 进入项目目录
%cd ViT_Reproduce

# 安装依赖
!pip install -q torch torchvision numpy Pillow matplotlib tqdm scipy timm

# 设置 Python 路径
import sys
sys.path.append('/content/ViT_Reproduce/base')

# 创建必要的目录
import os
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('data', exist_ok=True)
```

## 方法二：直接上传文件

### 步骤 1: 上传文件到 Colab

1. 在 Colab 中，点击左侧的文件图标
2. 上传 `base` 文件夹中的所有 `.py` 文件：
   - `model.py`
   - `train.py`
   - `eval.py`
   - `eval_cifar10.py`
   - `utils.py`
   - `test_model.py`

### 步骤 2: 安装依赖和设置

```python
# 安装依赖
!pip install -q torch torchvision numpy Pillow matplotlib tqdm scipy timm

# 设置路径（如果文件上传到 /content）
import sys
sys.path.append('/content')

# 创建目录
import os
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('data', exist_ok=True)
```

## 快速开始

### 1. 测试模型

```python
from model import vit_base_patch16_224
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建模型
model = vit_base_patch16_224(num_classes=10)
model = model.to(device)

# 测试
dummy_input = torch.randn(2, 3, 224, 224).to(device)
output = model(dummy_input)
print(f'Output shape: {output.shape}')
```

### 2. 使用预训练权重评估 CIFAR-10

```python
!python eval_cifar10.py --pretrained --model vit_base --batch_size 64
```

### 3. 训练模型（使用预训练权重）

```python
# 首先准备数据集（上传或使用现有数据）
# 然后运行训练
!python train.py \
    --data_dir ./data \
    --model vit_base \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --pretrained \
    --save_dir ./checkpoints
```

## Colab 配置建议

### 启用 GPU

1. 点击 **Runtime** → **Change runtime type**
2. 选择 **GPU** (T4 或更高)
3. 点击 **Save**

### 检查 GPU

```python
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
```

### 挂载 Google Drive（可选）

如果需要访问 Drive 中的文件：

```python
from google.colab import drive
drive.mount('/content/drive')

# 然后可以使用 Drive 中的文件
# 例如：--data_dir /content/drive/MyDrive/datasets/cifar10
```

## 完整示例

### 示例 1: CIFAR-10 评估

```python
# 1. 安装依赖
!pip install -q torch torchvision numpy Pillow matplotlib tqdm scipy timm

# 2. 克隆或上传项目（根据你的情况选择）
# !git clone https://github.com/yourusername/ViT_Reproduce.git
# %cd ViT_Reproduce

# 3. 设置路径
import sys
sys.path.append('/content/ViT_Reproduce/base')  # 或你的路径

# 4. 运行评估
!python eval_cifar10.py --pretrained --model vit_base --batch_size 64
```

### 示例 2: 训练自定义数据集

```python
# 1. 上传数据集
from google.colab import files
# 上传数据集 zip 文件，然后解压
# !unzip dataset.zip -d data/

# 2. 训练
!python train.py \
    --data_dir ./data \
    --model vit_base \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001 \
    --pretrained \
    --save_dir ./checkpoints
```

### 示例 3: 下载训练结果

```python
from google.colab import files

# 下载最佳模型
files.download('checkpoints/best.pth')

# 下载训练历史
files.download('checkpoints/history.json')
```

## 常见问题

### Q: 如何避免会话超时？

A: Colab 会在不活动后断开连接。可以：
- 定期运行代码保持活跃
- 使用 `!nvidia-smi` 检查 GPU
- 保存中间结果到 Google Drive

### Q: 如何保存训练进度？

A: 
```python
# 定期下载 checkpoint
from google.colab import files
files.download('checkpoints/latest.pth')
```

### Q: 内存不足怎么办？

A: 
- 减小 batch_size
- 使用较小的模型（vit_small）
- 清理缓存：`torch.cuda.empty_cache()`

### Q: 如何继续之前的训练？

A: 
```python
!python train.py \
    --data_dir ./data \
    --model vit_base \
    --resume ./checkpoints/latest.pth \
    --epochs 20
```

## 性能优化建议

1. **使用混合精度训练**（如果支持）：
   ```python
   from torch.cuda.amp import autocast, GradScaler
   # 在训练循环中使用 autocast
   ```

2. **调整 batch size**：
   - T4 GPU: batch_size=32-64
   - V100/A100: batch_size=64-128

3. **使用数据并行**（多 GPU）：
   ```python
   if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
   ```

## 注意事项

1. **会话限制**：免费版 Colab 有使用时间限制
2. **存储限制**：免费版约 15GB 存储空间
3. **GPU 限制**：免费版 GPU 使用有时间限制
4. **数据持久化**：Colab 重启后数据会丢失，需要重新上传或从 Drive 加载

## 快速参考

```python
# 完整设置脚本
!pip install -q torch torchvision numpy Pillow matplotlib tqdm scipy timm
import sys
sys.path.append('/content/ViT_Reproduce/base')
import os
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 测试
from model import vit_base_patch16_224
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vit_base_patch16_224(num_classes=10, pretrained=True).to(device)
print('Ready!')
```

## 更多资源

- [Colab 官方文档](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch Colab 教程](https://pytorch.org/tutorials/)
- 项目 README: `base/README.md`

