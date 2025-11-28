"""
配置文件 - GNSS干扰检测项目
包含所有超参数、路径配置和模型参数
"""
import os

# ============================================================================
# 数据路径配置
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'dataset_npy')

# 四个数据集
DATASETS = ['UTD', 'MCD', 'TGD', 'TGS']

# 类别配置
CLASSES = ['normal', 'attack', 'tracking_fail']
CLASS_TO_IDX = {'normal': 0, 'attack': 1, 'tracking_fail': 2}
IDX_TO_CLASS = {0: 'normal', 1: 'attack', 2: 'tracking_fail'}
NUM_CLASSES = 3

# 特征名称（9个tracking特征）
FEATURE_NAMES = [
    'I_P',           # Prompt 同相
    'Q_P',           # Prompt 正交
    'doppler',       # 多普勒频移
    'carrFreq',      # 载波频率
    'codePhase',     # 码相位
    'CN0fromSNR',    # 载噪比
    'pllLockIndicator',  # 锁相环指示器
    'fllLockIndicator',  # 锁频环指示器
    'dllDiscr'       # 码环鉴别器
]

# ============================================================================
# 数据集划分配置
# ============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# 数据处理配置
# ============================================================================
WINDOW_SIZE = 2000  # 时间窗口大小
NUM_FEATURES = 9    # 特征数量

# 数据归一化方法：'standard'(标准化) 或 'minmax'(归一化)
NORMALIZATION = 'standard'

# ============================================================================
# 训练通用配置
# ============================================================================
BATCH_SIZE = 64
NUM_WORKERS = 4  # DataLoader的工作进程数
DEVICE = 'cuda'  # 'cuda' 或 'cpu'

# ============================================================================
# LightGBM 配置
# ============================================================================
LIGHTGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': NUM_CLASSES,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4
}

LIGHTGBM_TRAIN_PARAMS = {
    'num_boost_round': 500,
    'early_stopping_rounds': 50,
    'verbose_eval': 50
}

# ============================================================================
# CNN 配置
# ============================================================================
CNN_CONFIG = {
    'input_channels': NUM_FEATURES,
    'num_classes': NUM_CLASSES,
    'dropout': 0.5,
}

CNN_TRAIN_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 20
}

# ============================================================================
# LSTM 配置
# ============================================================================
LSTM_CONFIG = {
    'input_size': NUM_FEATURES,
    'hidden_size': 128,
    'num_layers': 2,
    'num_classes': NUM_CLASSES,
    'dropout': 0.3,
    'bidirectional': True
}

LSTM_TRAIN_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 20
}

# ============================================================================
# 输出路径配置
# ============================================================================
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(OUTPUT_ROOT, 'models')
LOGS_DIR = os.path.join(OUTPUT_ROOT, 'logs')
FIGURES_DIR = os.path.join(OUTPUT_ROOT, 'figures')

# 创建输出目录
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# 实验配置
# ============================================================================
# 数据集使用模式：'single' 或 'mixed'
# 'single': 在单个数据集上训练
# 'mixed': 混合所有数据集训练
DATASET_MODE = 'mixed'  # 可以在运行时通过命令行参数修改

# 如果是single模式，使用哪个数据集
SINGLE_DATASET = 'UTD'  # 可选: 'UTD', 'MCD', 'TGD', 'TGS'
