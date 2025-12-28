# Lumi-Fusion

一个综合的低光照图像增强项目，集成了CLAHE、Gamma校正和Zero-DCE三种算法，并支持多种组合方式。

## 项目简介

Lumi-Fusion是一个用于低光照图像增强的研究项目，实现了三种经典的图像增强算法，并提供了7种不同的实验配置来评估这些算法的单独和组合使用效果。

## 算法说明

### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **描述**: 对比度受限的自适应直方图均衡化
- **特点**: 通过局部直方图均衡化来增强图像对比度，同时限制过度增强
- **参数**: 
  - `clip_limit`: 对比度限制（默认2.0）
  - `tile_grid_size`: 网格大小（默认8x8）

### 2. Gamma校正
- **描述**: 非线性亮度调整方法
- **特点**: 通过调整gamma值来改变图像的整体亮度
- **参数**:
  - `gamma`: gamma值，>1变亮，<1变暗（默认2.2）

### 3. Zero-DCE (Zero-Reference Deep Curve Estimation)
- **描述**: 基于深度学习的零参考低光照图像增强方法
- **特点**: 不需要参考图像进行训练，通过曲线估计实现自适应增强
- **模型**: 需要预训练的Zero-DCE模型（已包含在snapshots/目录中）

## 实验配置

项目支持以下7种实验配置：

| ID | 实验名称 | 方法组合 | 描述 |
|---|---|---|---|
| 1 | CLAHE | CLAHE | 仅使用CLAHE算法 |
| 2 | Gamma | Gamma | 仅使用Gamma校正 |
| 3 | Zero-DCE | Zero-DCE | 仅使用Zero-DCE模型 |
| 4 | CLAHE_Zero-DCE | CLAHE → Zero-DCE | 先CLAHE后Zero-DCE |
| 5 | Gamma_Zero-DCE | Gamma → Zero-DCE | 先Gamma后Zero-DCE |
| 6 | Zero-DCE_Gamma | Zero-DCE → Gamma | 先Zero-DCE后Gamma |
| 7 | CLAHE_Zero-DCE_Gamma | CLAHE → Zero-DCE → Gamma | 三种算法组合使用 |

## 评估指标

项目使用以下三个指标来评估增强效果：

1. **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，衡量图像质量
2. **SSIM (Structural Similarity Index)**: 结构相似性指数，衡量图像结构相似度
3. **信息熵 (Entropy)**: 衡量图像的细节和信息丰富程度

## 数据集

项目使用Zero-DCE数据集，数据目录结构如下：

```
data/
├── train_data/          # 训练数据
│   └── *.jpg
├── test_data/           # 测试数据
│   ├── DICM/
│   │   └── *.jpg
│   └── LIME/
│       └── *.bmp
└── result/              # 结果输出目录
```

## 环境配置

### 系统要求
- Python >= 3.7
- CUDA支持的GPU（可选，CPU也可运行但速度较慢）

### 安装步骤

1. 克隆项目（或下载项目文件）

2. 安装依赖：
```bash
pip install -r requirements.txt
```

注意：如果使用GPU，请根据你的CUDA版本安装对应的PyTorch版本：
- CUDA 10.2: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu102`
- CUDA 11.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu111`
- CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

对于Apple Silicon (M1/M2) Mac，PyTorch会自动使用MPS后端。

## 使用方法

### 1. 训练Zero-DCE模型

```bash
python lowlight_train.py \
    --lowlight_images_path data/train_data/ \
    --lr 0.0001 \
    --num_epochs 200 \
    --train_batch_size 8 \
    --snapshots_folder snapshots/
```

### 2. 运行单个实验

```bash
# 运行实验1 (CLAHE)
python test_experiments.py --experiment_ids 1

# 运行实验3 (Zero-DCE)
python test_experiments.py --experiment_ids 3

# 运行实验4 (CLAHE + Zero-DCE)
python test_experiments.py --experiment_ids 4
```

### 3. 运行所有实验

```bash
python test_experiments.py --run_all
```

### 4. 自定义参数

```bash
python test_experiments.py \
    --run_all \
    --clahe_clip_limit 3.0 \
    --clahe_tile_size 8 \
    --gamma_value 2.5 \
    --output_dir data/my_results/ \
    --save_results_csv my_results.csv
```

### 5. 参数说明

**主要参数**：
- `--test_data_path`: 测试数据路径（默认: `data/test_data/`）
- `--output_dir`: 结果输出目录（默认: `data/experiment_results/`）
- `--model_path`: Zero-DCE模型路径（默认: `snapshots/Epoch99.pth`）
- `--experiment_ids`: 要运行的实验ID列表（例如: `1 2 3`）
- `--run_all`: 运行所有7个实验

**算法参数**：
- `--clahe_clip_limit`: CLAHE对比度限制（默认: 2.0）
- `--clahe_tile_size`: CLAHE网格大小（默认: 8）
- `--gamma_value`: Gamma校正值（默认: 2.2）

**输出参数**：
- `--save_results_csv`: 保存CSV格式结果的文件路径
- `--save_results_json`: 保存JSON格式结果的文件路径

## 项目结构

```
Lumi-Fusion/
├── data/                    # 数据目录
│   ├── train_data/         # 训练数据
│   ├── test_data/          # 测试数据
│   └── result/             # 结果输出
├── snapshots/              # 模型检查点
│   ├── Epoch0.pth
│   ├── Epoch1.pth
│   └── Epoch99.pth
├── dataloader.py           # 数据加载器
├── model.py                # Zero-DCE模型定义
├── Myloss.py               # 损失函数
├── enhancement.py          # 增强算法实现（CLAHE, Gamma）
├── metrics.py              # 评估指标实现（PSNR, SSIM, Entropy）
├── utils.py                # 工具函数
├── lowlight_train.py       # 训练脚本
├── lowlight_test.py        # 原始测试脚本（仅Zero-DCE）
├── test_experiments.py     # 统一实验测试脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 本文件
```

## 结果说明

运行实验后，结果将保存在以下位置：

1. **增强后的图像**: `data/experiment_results/{实验名称}/`
2. **指标结果**: 
   - CSV格式: `experiment_results.csv`
   - JSON格式: `experiment_results.json`

结果文件包含以下字段：
- `experiment_id`: 实验ID
- `experiment_name`: 实验名称
- `num_images`: 处理的图像数量
- `avg_PSNR`: 平均PSNR值
- `avg_SSIM`: 平均SSIM值
- `avg_Entropy`: 平均信息熵值
- `std_PSNR`: PSNR标准差
- `std_SSIM`: SSIM标准差
- `std_Entropy`: 信息熵标准差

## 示例输出

```
============================================================
Running Experiment 1: CLAHE
Description: 仅使用CLAHE算法
============================================================

Found 88 images in 2 datasets
Processing [1/88]: 01.jpg
Processing [2/88]: 02.jpg
...

Experiment 1 Results:
  Average PSNR: 15.2345 ± 2.1234
  Average SSIM: 0.6789 ± 0.1234
  Average Entropy: 7.4567 ± 0.2345

============================================================
Summary of All Experiments
============================================================
Experiment                  PSNR        SSIM        Entropy    
------------------------------------------------------------
CLAHE                      15.2345     0.6789      7.4567
Gamma                      14.5678     0.6543      7.2345
Zero-DCE                   16.7890     0.7234      7.6789
...
============================================================
```

## 注意事项

1. **模型文件**: 确保Zero-DCE模型文件（`snapshots/Epoch99.pth`）存在，否则包含Zero-DCE的实验将无法运行。

2. **内存使用**: Zero-DCE模型需要一定的GPU/CPU内存。如果内存不足，可以减少batch size或使用CPU模式。

3. **图像格式**: 支持的图像格式包括JPG、PNG、BMP等常见格式。

4. **设备兼容性**: 代码会自动检测并使用可用的设备（CUDA > MPS > CPU）。

## 引用

如果本项目对您的研究有帮助，请引用相关论文：

- **Zero-DCE**: Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation (CVPR 2021)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Gamma Correction**: 经典的图像亮度调整方法

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交Issue。

## 更新日志

### v1.0.0
- 实现了CLAHE、Gamma和Zero-DCE三种增强算法
- 支持7种实验配置
- 实现了PSNR、SSIM和信息熵评估指标
- 提供了统一的实验测试脚本

