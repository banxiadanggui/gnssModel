# 快速开始指南

## 5分钟快速上手

### 第一步：环境配置

```bash
# 方法1：使用自动配置脚本（推荐）
setup_environment.bat

# 方法2：手动配置
conda env create -f environment.yml
conda activate gnss_ml

# 方法3：使用pip
conda create -n gnss_ml python=3.10 -y
conda activate gnss_ml
pip install -r requirements.txt
```

### 第二步：训练第一个模型

```bash
# 激活环境
conda activate gnss_ml

# 训练LightGBM（最快，适合快速测试）
python src\train.py --model lightgbm --mode mixed

# 训练CNN
python src\train.py --model cnn --mode mixed --batch_size 64

# 训练LSTM
python src\train.py --model lstm --mode mixed --batch_size 32

# 训练所有模型
python src\train.py --model all --mode mixed
```

### 第三步：评估模型

```bash
# 评估训练好的模型
python src\evaluate.py --model lightgbm --mode mixed --save_predictions
python src\evaluate.py --model cnn --mode mixed --save_predictions
python src\evaluate.py --model lstm --mode mixed --save_predictions
```

### 第四步：查看结果

训练完成后，结果保存在 `results/` 目录：

- **模型文件**: `results/models/`
- **可视化图表**: `results/figures/`
- **评估报告**: `results/logs/`

## 常用命令速查

### 使用批处理脚本

```bash
# 训练单个数据集
scripts\train_single_dataset.bat lightgbm UTD
scripts\train_single_dataset.bat cnn MCD
scripts\train_single_dataset.bat lstm TGD

# 训练混合数据集
scripts\train_mixed_dataset.bat lightgbm
scripts\train_mixed_dataset.bat cnn
scripts\train_mixed_dataset.bat lstm

# 训练所有模型
scripts\train_all_models.bat mixed
scripts\train_all_models.bat single UTD

# 评估模型
scripts\evaluate_model.bat cnn mixed
scripts\evaluate_model.bat lstm single UTD
```

### 训练参数调整

```bash
# 调整批次大小（内存不足时）
python src\train.py --model cnn --mode mixed --batch_size 32

# 使用不同的归一化方法
python src\train.py --model lstm --mode mixed --normalize minmax

# 训练单个数据集
python src\train.py --model cnn --mode single --dataset UTD

# 设置随机种子
python src\train.py --model lightgbm --mode mixed --seed 123
```

## 项目结构一览

```
beidou/
├── src/                    # Python源代码
│   ├── train.py           # 训练脚本（主入口）
│   ├── evaluate.py        # 评估脚本
│   ├── config.py          # 配置文件
│   ├── dataset.py         # 数据加载
│   ├── utils.py           # 工具函数
│   └── models/            # 模型定义
├── scripts/               # 批处理脚本和示例
├── data/                  # 数据目录
│   └── dataset_npy/      # NPY格式数据集
├── results/               # 训练结果输出
│   ├── models/           # 保存的模型
│   ├── figures/          # 可视化图表
│   └── logs/             # 日志和报告
└── README.md             # 完整文档
```

## 数据集说明

### 四个数据集
- **UTD**: University of Texas Dallas
- **MCD**: Multi-Constellation Dataset
- **TGD**: Two-GNSS Dataset
- **TGS**: Two-GNSS Spoofing

### 三个类别
- **normal**: 正常信号
- **attack**: 干扰/欺骗信号
- **tracking_fail**: 跟踪失败

### 数据格式
- **形状**: (2000, 9) - 2000个时间步 × 9个特征
- **特征**: I_P, Q_P, doppler, carrFreq, codePhase, CN0fromSNR, pllLockIndicator, fllLockIndicator, dllDiscr

## 模型对比

| 模型 | 优势 | 训练速度 | 适用场景 |
|------|------|----------|----------|
| **LightGBM** | 快速baseline，特征重要性 | ⚡⚡⚡ | 快速原型，特征分析 |
| **CNN** | 提取局部特征 | ⚡⚡ | 时间序列模式识别 |
| **LSTM** | 捕获长期依赖 | ⚡ | 序列建模 |

## 常见问题

### Q: 训练很慢怎么办？
A:
```bash
# 从LightGBM开始（最快）
python src\train.py --model lightgbm --mode mixed

# 减小批次大小
python src\train.py --model cnn --mode mixed --batch_size 32
```

### Q: 内存不足？
A:
```bash
# 训练单个数据集而不是混合
python src\train.py --model cnn --mode single --dataset UTD

# 减小批次大小
python src\train.py --model lstm --mode mixed --batch_size 16
```

### Q: 如何使用GPU？
A: 如果已安装CUDA版本的PyTorch，程序会自动使用GPU。检查：
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```

### Q: Windows下num_workers错误？
A: Windows建议设置为0：
```bash
python src\train.py --model cnn --mode mixed --num_workers 0
```

## 下一步

1. **查看详细文档**: [README.md](README.md)
2. **训练指南**: [scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)
3. **代码示例**: [scripts/example_usage.py](scripts/example_usage.py)
4. **修改超参数**: 编辑 `src/config.py`

## 获取帮助

```bash
# 查看命令行帮助
python src\train.py --help
python src\evaluate.py --help
```

---

**开始训练你的第一个模型吧！** 🚀

```bash
conda activate gnss_ml
python src\train.py --model lightgbm --mode mixed
```
