# GNSS干扰检测项目

基于GNSS信号跟踪数据的干扰和欺骗检测系统，包含完整的数据处理和机器学习训练管线。

## 项目结构

```
beidou/
├── data/                          # 数据目录
│   ├── dataset_npy/              # NPY格式数据集（用于ML训练）
│   │   ├── UTD/                  # UTD数据集
│   │   ├── MCD/                  # MCD数据集
│   │   ├── TGD/                  # TGD数据集
│   │   └── TGS/                  # TGS数据集
│   ├── processedMAT/             # 处理后的MAT文件
│   ├── share.mat/                # 共享MAT数据
│   │   └── final_mat/           # 最终MAT结果
│   └── signalData/               # 原始信号数据
├── src/                          # Python源代码
│   ├── config.py                 # 配置文件
│   ├── utils.py                  # 工具函数
│   ├── dataset.py                # 数据加载模块
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   └── models/                   # 模型定义
│       ├── lightgbm_model.py     # LightGBM模型
│       ├── cnn_model.py          # CNN模型
│       └── lstm_model.py         # LSTM模型
├── scripts/                      # 脚本和工具
│   ├── TRAINING_GUIDE.md         # 训练指南
│   ├── train_single_dataset.bat  # 单数据集训练脚本
│   ├── train_mixed_dataset.bat   # 混合数据集训练脚本
│   ├── train_all_models.bat      # 训练所有模型脚本
│   └── evaluate_model.bat        # 模型评估脚本
├── results/                      # 训练结果
│   ├── models/                   # 保存的模型
│   ├── figures/                  # 可视化图表
│   └── logs/                     # 日志和报告
├── matlabFGI_ori/                # MATLAB原始代码
├── matlabFGI_shared/             # MATLAB共享代码（并行处理版本）
├── toolsCode/                    # 数据转换工具
└── README.md                     # 本文件
```

## 环境要求

### Python环境（机器学习）

使用conda管理Python环境：

```bash
# 创建环境
conda create -n gnss_ml python=3.10 -y

# 激活环境
conda activate gnss_ml

# 安装依赖
conda install numpy pandas scipy matplotlib seaborn scikit-learn h5py openpyxl jupyter -y
pip install lightgbm torch torchvision torchaudio
```

### MATLAB环境（信号处理）

- MATLAB R2016b或更高版本
- Signal Processing Toolbox
- Parallel Computing Toolbox（用于并行处理版本）

## 快速开始

### 1. GNSS信号处理（MATLAB）

#### 使用FGI-GSRx处理GNSS信号

```matlab
% 进入MATLAB工作目录
cd matlabFGI_shared

% 编辑配置文件
edit init_gps.m  % 或 init_galileo.m, init_bds.m

% 运行信号处理
gsrx('init_gps.m')
```

#### 批量并行处理

```matlab
% 使用并行处理版本
cd matlabFGI_shared
parallel_process_all_files  % 自动处理所有信号文件
```

### 2. 机器学习训练（Python）

#### 快速训练示例

```bash
# 激活环境
conda activate gnss_ml

# 训练单个模型（混合数据集）
python src\train.py --model lightgbm --mode mixed

# 训练所有模型
python src\train.py --model all --mode mixed --batch_size 64

# 评估模型
python src\evaluate.py --model cnn --mode mixed --save_predictions
```

#### 使用批处理脚本

```bash
# 训练单个数据集
scripts\train_single_dataset.bat lstm UTD

# 训练混合数据集
scripts\train_mixed_dataset.bat cnn

# 训练所有模型
scripts\train_all_models.bat mixed

# 评估模型
scripts\evaluate_model.bat lstm mixed
```

详细的训练指南请参考：[scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)

## 数据结构

### 输入数据格式

#### NPY数据集（用于ML训练）
- **位置**: `data/dataset_npy/`
- **格式**: NumPy数组 (.npy)
- **形状**: (2000, 9) - 2000个时间步 × 9个特征
- **采样率**: 1000 Hz
- **类别**:
  - `normal/` - 正常信号
  - `attack/` - 干扰/欺骗信号
  - `tracking_fail/` - 跟踪失败

#### 9个跟踪特征
1. `I_P` - 同相提示相关值
2. `Q_P` - 正交提示相关值
3. `doppler` - 多普勒频移
4. `carrFreq` - 载波频率
5. `codePhase` - 码相位
6. `CN0fromSNR` - 载噪比（从SNR计算）
7. `pllLockIndicator` - PLL锁定指示器
8. `fllLockIndicator` - FLL锁定指示器
9. `dllDiscr` - DLL鉴别器输出

### 数据集说明

项目使用4个数据集：

| 数据集 | 说明 | 样本数 |
|--------|------|--------|
| **UTD** | University of Texas Dallas数据集 | 见数据目录 |
| **MCD** | Multi-Constellation Dataset | 见数据目录 |
| **TGD** | Two-GNSS Dataset | 见数据目录 |
| **TGS** | Two-GNSS Spoofing dataset | 见数据目录 |

### 训练模式

- **single模式**: 在单个数据集上独立训练
- **mixed模式**: 在所有4个数据集的混合数据上训练

### 数据划分

- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

## 模型说明

### 1. LightGBM
- **类型**: 梯度提升树
- **优势**: 训练快速，适合快速baseline
- **特点**: 提供特征重要性分析
- **输入**: 18000维（2000×9展平）

### 2. 1D-CNN
- **类型**: 一维卷积神经网络
- **架构**: 3个卷积块 (64→128→256) + 全局平均池化
- **优势**: 提取时间序列局部特征
- **输入**: (2000, 9)

### 3. Bi-LSTM
- **类型**: 双向长短期记忆网络
- **架构**: 2层LSTM，隐藏层128
- **优势**: 捕获长期时间依赖关系
- **输入**: (2000, 9)

## 训练和评估

### 训练命令

```bash
# 基本用法
python src\train.py --model [lightgbm|cnn|lstm|all] --mode [single|mixed] --dataset [UTD|MCD|TGD|TGS]

# 示例
python src\train.py --model cnn --mode mixed --batch_size 64
python src\train.py --model lstm --mode single --dataset UTD --batch_size 32
python src\train.py --model all --mode mixed --batch_size 64
```

### 评估命令

```bash
# 基本用法
python src\evaluate.py --model [lightgbm|cnn|lstm] --mode [single|mixed] --dataset [UTD|MCD|TGD|TGS]

# 示例
python src\evaluate.py --model cnn --mode mixed --save_predictions
python src\evaluate.py --model lstm --mode single --dataset UTD
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 | 必需 |
| `--mode` | 数据模式 (single/mixed) | mixed |
| `--dataset` | 数据集名称 | UTD |
| `--batch_size` | 批次大小 | 64 |
| `--num_workers` | 数据加载进程数 | 0 |
| `--normalize` | 归一化方法 | standard |
| `--seed` | 随机种子 | 42 |

## 输出结果

### 训练输出

训练完成后，结果保存在 `results/` 目录：

**模型文件** (`results/models/`):
- `lightgbm_[dataset]_best.txt`
- `cnn_[dataset]_best.pth`
- `lstm_[dataset]_best.pth`

**可视化** (`results/figures/`):
- 训练曲线: `[model]_[dataset]_training_history.png`
- 混淆矩阵: `[model]_[dataset]_confusion_matrix.png`
- 特征重要性: `lightgbm_[dataset]_feature_importance.png`

**日志** (`results/logs/`):
- 分类报告: `[model]_[dataset]_classification_report.txt`
- 详细结果: `[model]_[dataset]_[timestamp]_results.json`

### 评估输出

评估时会生成：
- 混淆矩阵图
- 分类报告
- 预测结果 (如果使用 `--save_predictions`)

## 数据可视化

### MATLAB可视化

FGI-GSRx内置可视化功能，在配置文件中设置：

```matlab
% init_*.m 文件中
sys.plotSpectra = 1;        % 频谱图
sys.plotAcquisition = 1;    % 捕获结果
sys.plotTracking = 1;       % 跟踪结果
```

### Python可视化

训练过程自动生成：
- 训练/验证曲线
- 混淆矩阵
- 特征重要性（LightGBM）

## 数据转换工具

`toolsCode/` 目录包含MAT到CSV/NPY的转换工具：

```bash
# MAT转NPY
python toolsCode\mat_to_npy_converter.py

# MAT转CSV
python toolsCode\gnss_mat_to_csv.py

# CSV转Excel
python toolsCode\csv_to_excel.py
```

## 配置文件

### Python配置 (`src/config.py`)

包含所有超参数和路径配置：
- 数据路径
- 模型架构参数
- 训练参数（学习率、批次大小等）
- 文件路径配置

### MATLAB配置

- `init_gps.m` - GPS信号处理配置
- `init_galileo.m` - Galileo信号处理配置
- `init_bds.m` - BeiDou信号处理配置

## 常见问题

### Q: 训练时显存不足？
A: 减小batch_size参数：
```bash
python src\train.py --model cnn --mode mixed --batch_size 32
```

### Q: Windows下数据加载出错？
A: 设置num_workers=0：
```bash
python src\train.py --model lstm --mode mixed --num_workers 0
```

### Q: 如何使用GPU？
A: 安装CUDA版本的PyTorch，程序会自动检测并使用GPU

### Q: 如何修改模型参数？
A: 编辑 `src/config.py` 中的配置字典

## Git仓库

**Git 仓库**: 已初始化

## 参考文献

### FGI-GSRx
- 基于芬兰地理空间研究所的GNSS软件接收机
- 支持GPS、Galileo、BeiDou信号处理

## 作者和贡献

本项目用于GNSS干扰和欺骗检测研究。

## 许可证

请遵守相关开源许可证和学术规范。

---

& "C:\Users\harin\miniconda3\shell\condabin\conda-hook.ps1"
conda activate
conda activate gnss_ml  

**更新日期**: 2025-11-28

详细训练指南请参考: [scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)
