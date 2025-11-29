# 6分类CNN模型说明

## 概述

6分类模型将GNSS信号细分为6个类别，相比3分类模型，可以区分不同数据集来源的干扰类型。

## 分类方案

| 标签 | 类别名称 | 说明 |
|------|---------|------|
| 0 | normal | 正常GNSS信号 |
| 1 | tracking_fail | 跟踪失败 |
| 2 | attack_UTD | UTD数据集的干扰信号 |
| 3 | attack_TGS | TGS数据集的干扰信号 |
| 4 | attack_TGD | TGD数据集的干扰信号 |
| 5 | attack_MCD | MCD数据集的干扰信号 |

## 模型架构

### CNN 6分类模型特点

1. **增强的网络深度**：
   - 4层卷积块（相比3分类的3层）
   - 更深的全连接层（512 → 256 → 128 → 6）

2. **结构细节**：
   ```
   输入: (Batch, 2000, 9)
   ↓
   Conv1D(9→64) → BN → ReLU → MaxPool
   Conv1D(64→128) → BN → ReLU → MaxPool
   Conv1D(128→256) → BN → ReLU → MaxPool
   Conv1D(256→512) → BN → ReLU → AdaptiveAvgPool
   ↓
   Flatten: (Batch, 512)
   ↓
   FC(512→256) → ReLU → Dropout(0.5)
   FC(256→128) → ReLU → Dropout(0.25)
   FC(128→6)
   ↓
   输出: (Batch, 6)
   ```

3. **参数量**：约 1.5M 参数

## 使用方法

### 1. 训练模型

```bash
conda activate gnss_ml
python scripts/train_cnn_6class.py
```

### 2. 训练配置

配置文件：`src/config_6class.py`

主要参数：
```python
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 20
```

### 3. 输出文件

训练完成后会生成：
- `results/models/cnn_6class_best.pth` - 最佳模型权重
- `results/figures/cnn_6class_training_history.png` - 训练曲线
- `results/figures/cnn_6class_confusion_matrix.png` - 混淆矩阵
- `results/logs/cnn_6class_classification_report.txt` - 分类报告
- `results/logs/cnn_6class_*_results.json` - 训练指标

## 与3分类模型对比

| 特性 | 3分类模型 | 6分类模型 |
|------|----------|----------|
| 类别数 | 3 | 6 |
| 卷积层 | 3层 | 4层 |
| 全连接层 | 256→128→3 | 512→256→128→6 |
| 参数量 | ~0.5M | ~1.5M |
| 训练难度 | 较低 | 较高 |
| 区分能力 | 基础 | 精细 |

## 应用场景

### 适合3分类的场景：
- 快速检测是否有干扰
- 实时系统对计算资源要求高
- 只需要知道有无干扰，不关心干扰来源

### 适合6分类的场景：
- 需要分析不同干扰类型的特征
- 研究不同干扰源的影响
- 训练数据充足，可以支持细粒度分类
- 对干扰来源溯源有需求

## 预期性能

基于数据集特性，预期性能：

- **总体准确率**：85-95%
- **Normal类**：95%+ （数据充足）
- **Tracking_fail类**：80-90%
- **Attack类**：80-90% （取决于数据集间差异）

不同attack子类的区分难度取决于：
1. 各数据集干扰信号的特征差异
2. 每个数据集的样本数量
3. 干扰信号的多样性

## 数据准备建议

为了获得最佳性能：

1. **数据平衡**：
   - 确保6个类别样本数量相对均衡
   - 可以考虑使用过采样/欠采样

2. **数据质量**：
   - 检查attack类是否有明显区分特征
   - 如果4个数据集的干扰信号特征相似，6分类可能退化为3分类

3. **数据增强**：
   - 对于样本少的类别，可以考虑数据增强
   - 例如：添加噪声、时间偏移等

## 代码文件说明

```
src/
├── config_6class.py          # 6分类配置
├── dataset_6class.py         # 6分类数据加载
└── models/
    └── cnn_6class_model.py   # 6分类CNN模型

scripts/
└── train_cnn_6class.py       # 训练脚本
```

## 注意事项

1. **显存需求**：
   - 由于模型更大，需要约 4GB 显存
   - 如果显存不足，可以减小batch_size

2. **训练时间**：
   - 相比3分类模型，训练时间增加约30-50%
   - GPU训练：约30-60分钟
   - CPU训练：约3-5小时

3. **过拟合风险**：
   - 6分类模型更容易过拟合
   - 已使用Dropout、Early Stopping等技术缓解
   - 建议监控训练/验证曲线

## 进一步优化

如果性能不理想，可以尝试：

1. **模型调整**：
   - 增加/减少卷积层
   - 调整dropout概率
   - 尝试不同的学习率

2. **数据策略**：
   - 数据增强
   - 类别平衡处理
   - 特征工程

3. **训练策略**：
   - 使用学习率warmup
   - 调整early stopping patience
   - 使用类别权重处理不平衡数据

## 问题排查

### 如果准确率很低（<70%）：
1. 检查数据是否正确加载
2. 检查标签是否正确分配
3. 检查是否有数据泄漏
4. 尝试降低模型复杂度

### 如果某些类别表现很差：
1. 检查该类别样本数量
2. 查看混淆矩阵，了解误分类模式
3. 考虑该类别是否与其他类别特征太相似

### 如果过拟合严重：
1. 增加Dropout
2. 增加数据增强
3. 减小模型规模
4. 早停patience设置更小
