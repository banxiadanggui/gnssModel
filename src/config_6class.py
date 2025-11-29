"""
6分类配置文件 - GNSS干扰检测项目
将干扰细分为4种数据集来源：UTD, TGS, TGD, MCD
"""
import os

# ============================================================================
# 数据路径配置
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'dataset_npy')

# 四个数据集
DATASETS = ['UTD', 'MCD', 'TGD', 'TGS']

# ============================================================================
# 6分类配置
# ============================================================================
# 新的6分类标签：
# - normal: 正常信号
# - tracking_fail: 跟踪失败
# - attack_UTD: UTD数据集的干扰
# - attack_TGS: TGS数据集的干扰
# - attack_TGD: TGD数据集的干扰
# - attack_MCD: MCD数据集的干扰

CLASSES_6 = ['normal', 'tracking_fail', 'attack_UTD', 'attack_TGS', 'attack_TGD', 'attack_MCD']

CLASS_TO_IDX_6 = {
    'normal': 0,
    'tracking_fail': 1,
    'attack_UTD': 2,
    'attack_TGS': 3,
    'attack_TGD': 4,
    'attack_MCD': 5
}

IDX_TO_CLASS_6 = {v: k for k, v in CLASS_TO_IDX_6.items()}
NUM_CLASSES_6 = 6

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
# CNN 6分类配置
# ============================================================================
CNN_6CLASS_CONFIG = {
    'input_channels': NUM_FEATURES,
    'num_classes': NUM_CLASSES_6,
    'dropout': 0.5,
}

CNN_6CLASS_TRAIN_CONFIG = {
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
