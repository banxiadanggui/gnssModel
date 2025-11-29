"""
6分类数据集加载模块
区分不同数据集来源的干扰：UTD, TGS, TGD, MCD
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import config_6class as config

class GNSSDataset6Class(Dataset):
    """GNSS干扰检测6分类数据集（PyTorch格式）"""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array, shape (N, 2000, 9)
            labels: numpy array, shape (N,) - 6分类标签
            transform: 数据转换函数
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def load_6class_dataset(data_root=config.DATA_ROOT):
    """
    加载6分类数据集

    区分规则：
    - normal和tracking_fail: 所有数据集共享
    - attack: 根据数据集来源区分为4类

    Args:
        data_root: 数据根目录

    Returns:
        data: numpy array, shape (N, 2000, 9)
        labels: numpy array, shape (N,) - 6分类标签
        sample_info: 样本信息列表
    """
    data_list = []
    labels_list = []
    sample_info = []

    print("\n" + "=" * 80)
    print("加载6分类数据集")
    print("=" * 80)
    print("分类方案：")
    print("  0: normal         - 正常信号")
    print("  1: tracking_fail  - 跟踪失败")
    print("  2: attack_UTD     - UTD数据集干扰")
    print("  3: attack_TGS     - TGS数据集干扰")
    print("  4: attack_TGD     - TGD数据集干扰")
    print("  5: attack_MCD     - MCD数据集干扰")
    print("=" * 80 + "\n")

    class_counts = {i: 0 for i in range(config.NUM_CLASSES_6)}

    for dataset_name in config.DATASETS:
        dataset_path = Path(data_root) / dataset_name

        if not dataset_path.exists():
            print(f"警告: 数据集路径不存在: {dataset_path}")
            continue

        print(f"处理数据集: {dataset_name}")
        print("-" * 60)

        # 处理每个原始类别
        for orig_class in ['normal', 'attack', 'tracking_fail']:
            class_dir = dataset_path / orig_class

            if not class_dir.exists():
                print(f"  警告: 类别目录不存在 {class_dir}")
                continue

            # 确定6分类标签
            if orig_class == 'normal':
                new_label = config.CLASS_TO_IDX_6['normal']
                new_class_name = 'normal'
            elif orig_class == 'tracking_fail':
                new_label = config.CLASS_TO_IDX_6['tracking_fail']
                new_class_name = 'tracking_fail'
            elif orig_class == 'attack':
                # 根据数据集来源区分attack
                new_class_name = f'attack_{dataset_name}'
                new_label = config.CLASS_TO_IDX_6[new_class_name]
            else:
                continue

            # 加载该类别的所有npy文件
            npy_files = sorted(class_dir.glob('*.npy'))

            for npy_file in npy_files:
                try:
                    # 加载数据并转换为float32
                    data = np.load(npy_file).astype(np.float32)

                    # 验证数据形状
                    if data.shape != (config.WINDOW_SIZE, config.NUM_FEATURES):
                        print(f"  警告: {npy_file.name} 形状不正确 {data.shape}, 跳过")
                        continue

                    data_list.append(data)
                    labels_list.append(new_label)
                    sample_info.append({
                        'dataset': dataset_name,
                        'original_class': orig_class,
                        'new_class': new_class_name,
                        'label': new_label,
                        'file': str(npy_file)
                    })
                    class_counts[new_label] += 1

                except Exception as e:
                    print(f"  错误: 加载 {npy_file.name} 失败: {e}")
                    continue

            print(f"  {orig_class:15s} -> {new_class_name:20s}: {len([s for s in sample_info if s['new_class'] == new_class_name and s['dataset'] == dataset_name])} 样本")

        print()

    # 转换为numpy数组
    data = np.array(data_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # 打印统计信息
    print("=" * 80)
    print("数据加载完成 - 样本分布：")
    print("=" * 80)
    for idx, class_name in config.IDX_TO_CLASS_6.items():
        count = class_counts[idx]
        percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
        print(f"  {class_name:20s} (标签{idx}): {count:6d} 样本 ({percentage:5.2f}%)")
    print("-" * 80)
    print(f"  总计:                  {len(labels):6d} 样本")
    print("=" * 80 + "\n")

    return data, labels, sample_info

def prepare_6class_data(normalize=True, normalization_method='standard'):
    """
    准备6分类训练数据

    Args:
        normalize: 是否归一化
        normalization_method: 归一化方法 ('standard' 或 'minmax')

    Returns:
        data_dict: 包含训练/验证/测试数据的字典
    """
    # 加载数据
    data, labels, sample_info = load_6class_dataset()

    # 数据集划分
    print("划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=labels
    )

    val_test_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_test_ratio,
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )

    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本\n")

    # 数据归一化
    if normalize:
        print(f"数据归一化 (方法: {normalization_method})...")

        # 重塑数据进行归一化: (N, T, F) -> (N*T, F)
        N_train, T, F = X_train.shape
        X_train_flat = X_train.reshape(-1, F)
        X_val_flat = X_val.reshape(-1, F)
        X_test_flat = X_test.reshape(-1, F)

        # 选择归一化器
        if normalization_method == 'standard':
            scaler = StandardScaler()
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"未知的归一化方法: {normalization_method}")

        # 拟合训练集并转换所有数据
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_val_flat = scaler.transform(X_val_flat)
        X_test_flat = scaler.transform(X_test_flat)

        # 重塑回原形状
        X_train = X_train_flat.reshape(N_train, T, F)
        X_val = X_val_flat.reshape(len(X_val), T, F)
        X_test = X_test_flat.reshape(len(X_test), T, F)

        print("  归一化完成\n")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'sample_info': sample_info
    }

def create_6class_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                              batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    创建6分类PyTorch DataLoader

    Args:
        X_train, X_val, X_test: 数据数组
        y_train, y_val, y_test: 标签数组
        batch_size: 批次大小
        num_workers: 数据加载进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建Dataset
    train_dataset = GNSSDataset6Class(X_train, y_train)
    val_dataset = GNSSDataset6Class(X_val, y_val)
    test_dataset = GNSSDataset6Class(X_test, y_test)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    print(f"DataLoader创建完成:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")

    return train_loader, val_loader, test_loader
