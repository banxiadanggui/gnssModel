# GNSS干扰检测模型训练指南

## 环境准备

首先激活conda环境：
```bash
conda activate gnss_ml
```

## 快速开始

### 使用批处理脚本（推荐）

#### 1. 训练单个数据集
```bash
# 在UTD数据集上训练LightGBM
scripts\train_single_dataset.bat lightgbm UTD

# 在MCD数据集上训练CNN
scripts\train_single_dataset.bat cnn MCD

# 在TGD数据集上训练LSTM
scripts\train_single_dataset.bat lstm TGD
```

#### 2. 训练混合数据集
```bash
# 在混合数据集上训练LightGBM
scripts\train_mixed_dataset.bat lightgbm

# 在混合数据集上训练CNN
scripts\train_mixed_dataset.bat cnn

# 在混合数据集上训练LSTM
scripts\train_mixed_dataset.bat lstm
```

#### 3. 训练所有模型
```bash
# 在混合数据集上训练所有模型
scripts\train_all_models.bat mixed

# 在UTD数据集上训练所有模型
scripts\train_all_models.bat single UTD
```

#### 4. 评估模型
```bash
# 评估混合数据集上的CNN模型
scripts\evaluate_model.bat cnn mixed

# 评估UTD数据集上的LSTM模型
scripts\evaluate_model.bat lstm single UTD
```

## 直接使用Python命令

### 训练模型

#### 训练单个模型（单个数据集）
```bash
# LightGBM on UTD
python src\train.py --model lightgbm --mode single --dataset UTD

# CNN on MCD
python src\train.py --model cnn --mode single --dataset MCD --batch_size 64

# LSTM on TGD
python src\train.py --model lstm --mode single --dataset TGD --batch_size 32
```

#### 训练单个模型（混合数据集）
```bash
# LightGBM on mixed datasets
python src\train.py --model lightgbm --mode mixed

# CNN on mixed datasets
python src\train.py --model cnn --mode mixed --batch_size 64

# LSTM on mixed datasets
python src\train.py --model lstm --mode mixed --batch_size 32
```

#### 训练所有模型
```bash
# 在混合数据集上训练所有模型
python src\train.py --model all --mode mixed --batch_size 64

# 在UTD数据集上训练所有模型
python src\train.py --model all --mode single --dataset UTD --batch_size 64
```

### 评估模型

```bash
# 评估LightGBM（混合数据集）
python src\evaluate.py --model lightgbm --mode mixed --save_predictions

# 评估CNN（UTD数据集）
python src\evaluate.py --model cnn --mode single --dataset UTD --save_predictions

# 评估LSTM（自定义模型路径）
python src\evaluate.py --model lstm --mode mixed --model_path results\models\lstm_mixed_best.pth
```

## 命令行参数说明

### train.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必需 | 模型类型: lightgbm, cnn, lstm, all |
| `--mode` | str | mixed | 数据模式: single, mixed |
| `--dataset` | str | UTD | 数据集名称: UTD, MCD, TGD, TGS |
| `--normalize` | str | standard | 归一化方法: standard, minmax, none |
| `--batch_size` | int | 64 | 批次大小（CNN/LSTM） |
| `--num_workers` | int | 0 | 数据加载进程数 |
| `--seed` | int | 42 | 随机种子 |

### evaluate.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必需 | 模型类型: lightgbm, cnn, lstm |
| `--model_path` | str | 自动 | 模型文件路径 |
| `--mode` | str | mixed | 数据模式: single, mixed |
| `--dataset` | str | UTD | 数据集名称: UTD, MCD, TGD, TGS |
| `--normalize` | str | standard | 归一化方法: standard, minmax, none |
| `--batch_size` | int | 64 | 批次大小（CNN/LSTM） |
| `--num_workers` | int | 0 | 数据加载进程数 |
| `--seed` | int | 42 | 随机种子 |
| `--save_predictions` | flag | False | 是否保存预测结果 |

## 输出文件说明

训练和评估完成后，结果会保存在以下位置：

### 模型文件
- `results/models/lightgbm_[dataset]_best.txt` - LightGBM模型
- `results/models/cnn_[dataset]_best.pth` - CNN模型
- `results/models/lstm_[dataset]_best.pth` - LSTM模型

### 可视化图表
- `results/figures/[model]_[dataset]_training_history.png` - 训练曲线（CNN/LSTM）
- `results/figures/[model]_[dataset]_confusion_matrix.png` - 混淆矩阵
- `results/figures/lightgbm_[dataset]_feature_importance.png` - 特征重要性（LightGBM）

### 评估报告
- `results/logs/[model]_[dataset]_classification_report.txt` - 分类报告
- `results/logs/[model]_[dataset]_[timestamp]_results.json` - 详细结果JSON

### 预测结果
- `results/logs/[model]_[dataset]_predictions.npz` - 预测结果（需要--save_predictions）

## 数据集说明

### 四个数据集
- **UTD**: University of Texas Dallas数据集
- **MCD**: Multi-Constellation Dataset
- **TGD**: Two-GNSS Dataset
- **TGS**: Two-GNSS Spoofing dataset

### 数据模式
- **single模式**: 在单个数据集上训练和测试
- **mixed模式**: 在所有4个数据集的混合数据上训练和测试

### 数据划分
- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

## 模型说明

### LightGBM
- 基于梯度提升的树模型
- 训练速度快，适合快速baseline
- 提供特征重要性分析
- 输入: 将(2000, 9)展平为18000维特征

### CNN
- 1D卷积神经网络
- 3个卷积块: 64 → 128 → 256
- 全局平均池化 + 全连接层
- 适合提取时间序列局部特征

### LSTM
- 双向LSTM网络
- 2层LSTM，隐藏层大小128
- 适合捕获长期时间依赖关系
- 梯度裁剪防止梯度爆炸

## 超参数配置

所有超参数在 `src/config.py` 中配置，包括：
- 模型架构参数
- 训练参数（学习率、批次大小等）
- 数据参数（窗口大小、特征数等）
- 路径配置

## 常见问题

### Q: 训练时内存不足怎么办？
A: 可以减小batch_size，例如从64改为32或16：
```bash
python src\train.py --model cnn --mode mixed --batch_size 32
```

### Q: 如何使用GPU加速？
A: 如果安装了CUDA版本的PyTorch，程序会自动使用GPU。检查GPU可用性：
```python
import torch
print(torch.cuda.is_available())
```

### Q: 如何修改模型超参数？
A: 编辑 `src/config.py` 文件中的对应配置字典（CNN_CONFIG, LSTM_CONFIG等）

### Q: Windows下num_workers应该设置为多少？
A: 建议设置为0，因为Windows的多进程支持与Linux不同，设置大于0可能导致错误

### Q: 如何查看训练历史？
A: 训练完成后，查看 `results/figures/` 下的训练曲线图，或者加载模型文件中保存的history字典

## 示例工作流

### 完整的训练和评估流程

```bash
# 1. 激活环境
conda activate gnss_ml

# 2. 在混合数据集上训练所有模型
python src\train.py --model all --mode mixed --batch_size 64

# 3. 评估每个模型
python src\evaluate.py --model lightgbm --mode mixed --save_predictions
python src\evaluate.py --model cnn --mode mixed --save_predictions
python src\evaluate.py --model lstm --mode mixed --save_predictions

# 4. 查看结果
# - 查看混淆矩阵: results/figures/
# - 查看分类报告: results/logs/
# - 加载预测结果: results/logs/*_predictions.npz
```

### 对比单个数据集和混合数据集

```bash
# 在UTD上训练
python src\train.py --model lstm --mode single --dataset UTD

# 在混合数据集上训练
python src\train.py --model lstm --mode mixed

# 对比结果
# 查看 results/figures/ 和 results/logs/ 中的对应文件
```
