# GNSS干扰检测项目总结

## 项目概述

本项目实现了基于机器学习的GNSS（GPS/Galileo/BeiDou）信号干扰和欺骗检测系统，包含完整的数据处理和模型训练管线。

**创建时间**: 2025-11-29
**Git仓库**: 已初始化

## 完成的工作

### 1. 核心代码模块 (src/)

#### 配置和工具
- ✅ `config.py` - 集中式配置文件
  - 数据路径配置
  - 模型超参数配置（LightGBM/CNN/LSTM）
  - 训练参数配置
  - 输出路径配置

- ✅ `utils.py` - 工具函数库
  - 随机种子设置
  - 混淆矩阵绘制
  - 训练历史可视化
  - 分类报告生成
  - 早停机制
  - 设备检测（GPU/CPU）
  - 模型参数统计

- ✅ `dataset.py` - 数据加载模块
  - 单数据集加载
  - 混合数据集加载
  - 数据归一化（Standard/MinMax）
  - 数据集划分（70/15/15）
  - PyTorch DataLoader创建

#### 模型实现 (src/models/)

- ✅ `lightgbm_model.py` - LightGBM模型
  - 梯度提升树分类器
  - 特征重要性分析
  - 早停训练
  - 完整的训练/评估流程

- ✅ `cnn_model.py` - 1D-CNN模型
  - 3层卷积网络 (64→128→256)
  - 批归一化和Dropout
  - 全局平均池化
  - 学习率调度
  - 训练历史记录

- ✅ `lstm_model.py` - Bi-LSTM模型
  - 双向LSTM架构
  - 梯度裁剪
  - 2层LSTM，隐藏层128
  - 完整的训练/评估流程

#### 主程序

- ✅ `train.py` - 训练主脚本
  - 命令行参数解析
  - 支持三种模型
  - 支持单数据集/混合数据集模式
  - 支持训练所有模型
  - 自动保存模型和结果

- ✅ `evaluate.py` - 评估脚本
  - 加载已保存的模型
  - 测试集评估
  - 生成混淆矩阵和报告
  - 可选的预测结果保存

### 2. 批处理脚本 (scripts/)

- ✅ `train_single_dataset.bat` - 单数据集训练
- ✅ `train_mixed_dataset.bat` - 混合数据集训练
- ✅ `train_all_models.bat` - 训练所有模型
- ✅ `evaluate_model.bat` - 模型评估
- ✅ `example_usage.py` - Python使用示例
  - 5个详细的使用示例
  - 演示自定义训练参数
  - 演示模型加载和预测

### 3. 文档

- ✅ `README.md` - 主文档
  - 项目结构说明
  - 环境配置指南
  - 数据结构文档
  - 模型说明
  - 使用示例
  - 常见问题

- ✅ `scripts/TRAINING_GUIDE.md` - 训练指南
  - 详细的命令行参数说明
  - 训练示例
  - 输出文件说明
  - 疑难解答

- ✅ `QUICKSTART.md` - 快速开始指南
  - 5分钟上手教程
  - 常用命令速查
  - 快速问题解决

- ✅ `PROJECT_SUMMARY.md` - 本文档

### 4. 环境配置

- ✅ `requirements.txt` - pip依赖列表
- ✅ `environment.yml` - conda环境配置
- ✅ `setup_environment.bat` - 自动环境配置脚本
- ✅ `.gitignore` - Git忽略规则

## 技术架构

### 数据流程
```
原始数据 (data/dataset_npy/)
    ↓
数据加载 (dataset.py)
    ↓
数据归一化 (StandardScaler)
    ↓
数据集划分 (train/val/test: 70/15/15)
    ↓
DataLoader创建
    ↓
模型训练 (train.py)
    ↓
模型评估 (evaluate.py)
    ↓
结果保存 (results/)
```

### 模型架构

#### LightGBM
- 输入: (N, 18000) - 展平后的特征
- 输出: 3类别分类
- 特点: 快速训练，特征重要性

#### CNN
```
输入 (B, 2000, 9)
    ↓ Transpose
(B, 9, 2000)
    ↓ Conv1d(9→64, k=7) + BN + ReLU + MaxPool
(B, 64, 1000)
    ↓ Conv1d(64→128, k=5) + BN + ReLU + MaxPool
(B, 128, 500)
    ↓ Conv1d(128→256, k=3) + BN + ReLU + MaxPool
(B, 256, 250)
    ↓ AdaptiveAvgPool1d
(B, 256, 1)
    ↓ Squeeze + FC(256→128→3)
(B, 3)
```

#### LSTM
```
输入 (B, 2000, 9)
    ↓ Bi-LSTM(2层, hidden=128)
(B, 2000, 256)
    ↓ 取最后时间步的hidden state
(B, 256)
    ↓ FC(256→128→3)
(B, 3)
```

## 数据集统计

### 四个数据集
| 数据集 | normal | attack | tracking_fail | 总计 |
|--------|--------|--------|---------------|------|
| UTD    | 1,660  | 344    | 2,832         | 4,836 |
| MCD    | 2,081  | 4,488  | 11            | 6,580 |
| TGD    | 1,438  | 2,673  | 3             | 4,114 |
| TGS    | 1,316  | 2,490  | 4             | 3,810 |
| **混合** | **6,495** | **9,995** | **2,850** | **19,340** |

### 混合数据集类别分布
- normal: 33.6%
- attack: 51.7%
- tracking_fail: 14.7%

### 数据划分（混合数据集）
- 训练集: 13,538 样本 (70%)
- 验证集: 2,901 样本 (15%)
- 测试集: 2,901 样本 (15%)

## 模型参数统计

| 模型 | 可训练参数 | 输入形状 | 输出形状 |
|------|-----------|----------|----------|
| LightGBM | ~数千棵树 | (N, 18000) | (N, 3) |
| CNN | 177,923 | (N, 2000, 9) | (N, 3) |
| LSTM | ~300K+ | (N, 2000, 9) | (N, 3) |

## 使用方式

### 命令行训练
```bash
# 训练单个模型
python src\train.py --model [lightgbm|cnn|lstm] --mode [single|mixed]

# 训练所有模型
python src\train.py --model all --mode mixed

# 评估模型
python src\evaluate.py --model [lightgbm|cnn|lstm] --mode [single|mixed]
```

### 批处理脚本
```bash
scripts\train_single_dataset.bat [模型] [数据集]
scripts\train_mixed_dataset.bat [模型]
scripts\train_all_models.bat [模式]
scripts\evaluate_model.bat [模型] [模式]
```

### Python编程
```python
from dataset import prepare_data, create_dataloaders
from models.lightgbm_model import train_lightgbm

# 准备数据
data_dict = prepare_data(dataset_mode='mixed')

# 训练模型
model, results = train_lightgbm(data_dict, dataset_mode='mixed')
```

## 输出文件

### 模型文件 (results/models/)
- `lightgbm_[dataset]_best.txt`
- `cnn_[dataset]_best.pth`
- `lstm_[dataset]_best.pth`

### 可视化 (results/figures/)
- 训练曲线: `[model]_[dataset]_training_history.png`
- 混淆矩阵: `[model]_[dataset]_confusion_matrix.png`
- 特征重要性: `lightgbm_[dataset]_feature_importance.png`

### 日志 (results/logs/)
- 分类报告: `[model]_[dataset]_classification_report.txt`
- 详细结果: `[model]_[dataset]_[timestamp]_results.json`
- 预测结果: `[model]_[dataset]_predictions.npz`

## 已知问题和修复

### ✅ 已修复: ReduceLROnPlateau verbose参数
- **问题**: PyTorch新版本不支持 `verbose` 参数
- **修复**: 从cnn_model.py、lstm_model.py和example_usage.py中移除该参数
- **影响文件**:
  - src/models/cnn_model.py
  - src/models/lstm_model.py
  - scripts/example_usage.py

## 配置要点

### Python环境
- Python 3.10
- PyTorch >= 2.0.0
- LightGBM >= 3.3.0
- scikit-learn >= 1.1.0
- 其他科学计算库

### 关键超参数

**LightGBM**:
- num_leaves: 31
- learning_rate: 0.05
- num_boost_round: 500

**CNN**:
- batch_size: 64
- learning_rate: 0.001
- dropout: 0.5
- epochs: 100

**LSTM**:
- hidden_size: 128
- num_layers: 2
- bidirectional: True
- batch_size: 64
- learning_rate: 0.001

## 下一步工作建议

### 可能的改进方向
1. **模型优化**
   - 超参数调优
   - 集成学习方法
   - 注意力机制

2. **数据增强**
   - 时间序列数据增强
   - 类别平衡处理

3. **特征工程**
   - 手工特征提取
   - 特征选择

4. **部署**
   - 模型量化
   - ONNX导出
   - 实时推理API

5. **可视化**
   - 实时训练监控（TensorBoard）
   - 更多分析图表

## 文件清单

### Python源代码 (9个文件)
```
src/
├── config.py
├── utils.py
├── dataset.py
├── train.py
├── evaluate.py
└── models/
    ├── lightgbm_model.py
    ├── cnn_model.py
    └── lstm_model.py
```

### 脚本和示例 (6个文件)
```
scripts/
├── train_single_dataset.bat
├── train_mixed_dataset.bat
├── train_all_models.bat
├── evaluate_model.bat
├── example_usage.py
└── TRAINING_GUIDE.md
```

### 文档 (4个文件)
```
├── README.md
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
└── requirements.txt
```

### 配置文件 (4个文件)
```
├── environment.yml
├── setup_environment.bat
└── .gitignore
```

**总计**: 23个新创建的文件

## 项目特点

### ✅ 优势
1. **完整的训练管线** - 从数据加载到模型评估
2. **灵活的配置** - 集中式配置管理
3. **多模型支持** - LightGBM、CNN、LSTM
4. **双模式训练** - 单数据集和混合数据集
5. **详细的文档** - README、训练指南、快速开始
6. **自动化脚本** - Windows批处理脚本
7. **代码示例** - 5个完整的使用示例
8. **可视化输出** - 训练曲线、混淆矩阵、特征重要性
9. **模块化设计** - 易于扩展和修改

### 🎯 适用场景
- GNSS干扰检测研究
- 时间序列分类任务
- 机器学习教学示例
- 快速原型开发

## 总结

本项目提供了一个完整的、开箱即用的GNSS干扰检测训练系统，包含：
- ✅ 3种机器学习模型
- ✅ 4个数据集支持
- ✅ 完整的训练和评估流程
- ✅ 详细的文档和示例
- ✅ 自动化脚本工具

用户可以通过简单的命令行或批处理脚本快速开始训练，也可以通过Python代码进行深度定制。

---

**开始使用**: 参考 [QUICKSTART.md](QUICKSTART.md)
**详细文档**: 参考 [README.md](README.md)
**训练指南**: 参考 [scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)
