# 6分类CNN模型快速开始

## 🚀 快速开始（3步）

### 步骤1：激活环境
```bash
conda activate gnss_ml
```

### 步骤2：训练模型
```bash
python scripts/train_cnn_6class.py
```

### 步骤3：查看结果
```bash
# 查看训练曲线
start results/figures/cnn_6class_training_history.png

# 查看混淆矩阵
start results/figures/cnn_6class_confusion_matrix.png

# 查看分类报告
type results/logs/cnn_6class_classification_report.txt
```

## 📊 预期输出示例

### 训练过程
```
============================================================
开始训练 6分类CNN 模型
============================================================
分类目标:
  0: normal
  1: tracking_fail
  2: attack_UTD
  3: attack_TGS
  4: attack_TGD
  5: attack_MCD
============================================================

使用GPU: NVIDIA GeForce RTX 4080 Laptop GPU
显存大小: 12.0 GB
已启用 cuDNN benchmark 优化

Epoch [1/100] Train Loss: 1.2345 Acc: 45.67% | Val Loss: 1.1234 Acc: 52.34%
Epoch [2/100] Train Loss: 1.0123 Acc: 58.23% | Val Loss: 0.9876 Acc: 63.45%
...
```

### 最终结果
```
============================================================
训练完成!
============================================================
测试集准确率: 0.8756
测试集精确率: 0.8689
测试集召回率: 0.8723
测试集F1分数: 0.8701
============================================================
```

## 🎯 6分类方案

```
┌─────────────────────────────────────────────┐
│          6分类GNSS干扰检测                  │
├─────────────────────────────────────────────┤
│                                             │
│  0: normal (正常)         ← 所有数据集共享  │
│  1: tracking_fail (失败)  ← 所有数据集共享  │
│                                             │
│  2: attack_UTD (UTD干扰)  ← 按数据集区分    │
│  3: attack_TGS (TGS干扰)  ← 按数据集区分    │
│  4: attack_TGD (TGD干扰)  ← 按数据集区分    │
│  5: attack_MCD (MCD干扰)  ← 按数据集区分    │
│                                             │
└─────────────────────────────────────────────┘
```

## 🔧 关键配置

### 模型配置 (config_6class.py)
```python
CNN_6CLASS_CONFIG = {
    'input_channels': 9,
    'num_classes': 6,
    'dropout': 0.5,
}

CNN_6CLASS_TRAIN_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 64,
}
```

### 如何调整

1. **减小batch size（如果显存不足）**：
   ```python
   # 在 config_6class.py 中修改
   BATCH_SIZE = 32  # 从64改为32
   ```

2. **调整训练轮数**：
   ```python
   CNN_6CLASS_TRAIN_CONFIG = {
       'epochs': 50,  # 减少训练时间
       ...
   }
   ```

3. **调整学习率**：
   ```python
   CNN_6CLASS_TRAIN_CONFIG = {
       'learning_rate': 0.0005,  # 更保守的学习率
       ...
   }
   ```

## 📁 生成的文件

训练完成后，会在`results/`目录生成：

```
results/
├── models/
│   └── cnn_6class_best.pth          # 最佳模型（忽略）
├── figures/
│   ├── cnn_6class_training_history.png  # 训练曲线
│   └── cnn_6class_confusion_matrix.png  # 混淆矩阵
└── logs/
    ├── cnn_6class_classification_report.txt
    └── cnn_6class_*_results.json
```

## ⏱️ 训练时间估计

| 硬件配置 | 预估时间 |
|---------|---------|
| RTX 4080 Laptop GPU | 30-45分钟 |
| RTX 3080 | 40-60分钟 |
| GTX 1080 Ti | 60-90分钟 |
| CPU (i7) | 3-5小时 |

## 🎓 理解训练过程

### 监控指标

1. **Train Loss 下降** → 模型在学习
2. **Val Loss 下降** → 模型泛化能力好
3. **Val Loss 上升** → 可能过拟合（会触发早停）
4. **Accuracy 稳定在高位** → 模型收敛

### 正常现象

- 前几个epoch accuracy快速上升
- 后期loss波动但整体下降
- 可能在50-80 epoch触发早停

### 异常情况

- Loss持续上升 → 学习率太大
- Accuracy停在很低水平 → 数据或模型问题
- 训练和验证差距很大 → 过拟合

## ❓ 常见问题

### Q: 显存不足怎么办？
A: 修改`config_6class.py`中的`BATCH_SIZE = 32`或更小

### Q: 训练太慢怎么办？
A:
1. 确认GPU正在使用（会显示GPU信息）
2. 减少epochs数量
3. 检查NUM_WORKERS设置

### Q: 如何使用训练好的模型？
A: 查看`scripts/example_usage.py`中的模型加载示例

### Q: 6分类和3分类哪个好？
A:
- **3分类**：快速、简单、准确率高
- **6分类**：细粒度、可分析不同干扰源、研究价值高

## 📈 性能优化建议

### 如果准确率低于80%：
1. 检查数据是否平衡
2. 增加训练轮数
3. 调整学习率
4. 尝试不同的dropout值

### 如果训练速度慢：
1. 增大batch size（如果显存允许）
2. 调整NUM_WORKERS
3. 确认GPU正在使用

### 如果过拟合：
1. 增加dropout (0.5 → 0.6)
2. 减少epochs
3. 使用数据增强

## 🔗 相关文档

- 详细说明：`docs/6CLASS_MODEL_README.md`
- 原始3分类模型：`src/models/cnn_model.py`
- 配置文件：`src/config_6class.py`
- 数据加载：`src/dataset_6class.py`
