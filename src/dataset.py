"""
数据集加载模块
支持单个数据集和混合数据集两种模式
"""
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import config

class GNSSDataset(Dataset):
    """GNSS干扰检测数据集（PyTorch格式）"""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array, shape (N, 2000, 9)
            labels: numpy array, shape (N,)
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

def load_single_dataset(dataset_name, data_root=config.DATA_ROOT):
    """
    加载单个数据集

    Args:
        dataset_name: 数据集名称 ('UTD', 'MCD', 'TGD', 'TGS')
        data_root: 数据根目录

    Returns:
        data: numpy array, shape (N, 2000, 9)
        labels: numpy array, shape (N,)
        sample_info: 样本信息列表
    """
    dataset_path = Path(data_root) / dataset_name

    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")

    data_list = []
    labels_list = []
    sample_info = []

    print(f"\n加载数据集: {dataset_name}")
    print("-" * 60)

    for class_name in config.CLASSES:
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            print(f"警告: 类别目录不存在 {class_dir}")
            continue

        # 获取该类别的所有npy文件
        npy_files = sorted(class_dir.glob('*.npy'))

        print(f"  {class_name:15s}: {len(npy_files):5d} 样本")

        for npy_file in npy_files:
            try:
                # 加载数据并转换为float32节省内存
                data = np.load(npy_file).astype(np.float32)

                # 验证数据形状
                if data.shape != (config.WINDOW_SIZE, config.NUM_FEATURES):
                    print(f"警告: {npy_file} 形状不正确 {data.shape}, 跳过")
                    continue

                data_list.append(data)
                labels_list.append(config.CLASS_TO_IDX[class_name])
                sample_info.append({
                    'dataset': dataset_name,
                    'class': class_name,
                    'file': str(npy_file)
                })

            except Exception as e:
                print(f"错误: 加载 {npy_file} 失败: {e}")
                continue

    print("-" * 60)

    if len(data_list) == 0:
        raise ValueError(f"数据集 {dataset_name} 中没有找到有效样本")

    data = np.array(data_list)  # shape: (N, 2000, 9)
    labels = np.array(labels_list)  # shape: (N,)

    print(f"总样本数: {len(data)}")
    print(f"数据形状: {data.shape}")
    print(f"标签分布: ", {config.IDX_TO_CLASS[i]: np.sum(labels == i) for i in range(config.NUM_CLASSES)})

    return data, labels, sample_info

def load_mixed_datasets(datasets=None, data_root=config.DATA_ROOT):
    """
    加载并混合多个数据集

    Args:
        datasets: 数据集名称列表，默认为所有数据集
        data_root: 数据根目录

    Returns:
        data: numpy array, shape (N, 2000, 9)
        labels: numpy array, shape (N,)
        sample_info: 样本信息列表
    """
    if datasets is None:
        datasets = config.DATASETS

    all_data = []
    all_labels = []
    all_info = []

    print("\n" + "="*60)
    print("加载混合数据集")
    print("="*60)

    for dataset_name in datasets:
        try:
            data, labels, sample_info = load_single_dataset(dataset_name, data_root)
            all_data.append(data)
            all_labels.append(labels)
            all_info.extend(sample_info)
        except Exception as e:
            print(f"警告: 跳过数据集 {dataset_name}: {e}")
            continue

    if len(all_data) == 0:
        raise ValueError("没有成功加载任何数据集")

    # 合并所有数据
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("\n" + "="*60)
    print("混合数据集统计:")
    print("="*60)
    print(f"总样本数: {len(data)}")
    print(f"数据形状: {data.shape}")
    print(f"标签分布: ")
    for i in range(config.NUM_CLASSES):
        count = np.sum(labels == i)
        percentage = count / len(labels) * 100
        print(f"  {config.IDX_TO_CLASS[i]:15s}: {count:5d} ({percentage:.1f}%)")
    print("="*60 + "\n")

    return data, labels, all_info

def normalize_data(X_train, X_val, X_test, method='standard'):
    """
    数据归一化

    Args:
        X_train, X_val, X_test: 训练、验证、测试数据
        method: 'standard' 或 'minmax'

    Returns:
        归一化后的数据和scaler
    """
    # 将数据reshape为 (N, 2000*9) 以便归一化
    N_train, T, F = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]

    X_train_flat = X_train.reshape(N_train, T * F)
    X_val_flat = X_val.reshape(N_val, T * F)
    X_test_flat = X_test.reshape(N_test, T * F)

    # 选择归一化方法
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"未知的归一化方法: {method}")

    # 在训练集上拟合scaler，并转换为float32节省内存
    X_train_norm = scaler.fit_transform(X_train_flat).astype(np.float32)
    X_val_norm = scaler.transform(X_val_flat).astype(np.float32)
    X_test_norm = scaler.transform(X_test_flat).astype(np.float32)

    # Reshape回原始形状
    X_train_norm = X_train_norm.reshape(N_train, T, F)
    X_val_norm = X_val_norm.reshape(N_val, T, F)
    X_test_norm = X_test_norm.reshape(N_test, T, F)

    return X_train_norm, X_val_norm, X_test_norm, scaler

def split_data(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
               random_seed=42, stratify=True):
    """
    划分训练、验证、测试集

    Args:
        data: numpy array
        labels: numpy array
        train_ratio, val_ratio, test_ratio: 划分比例
        random_seed: 随机种子
        stratify: 是否保持类别比例

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "划分比例之和必须为1"

    # 第一次划分：分离出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels if stratify else None
    )

    # 第二次划分：从剩余数据中分离出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=y_temp if stratify else None
    )

    print("\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(X_val)} 样本 ({val_ratio*100:.0f}%)")
    print(f"  测试集: {len(X_test)} 样本 ({test_ratio*100:.0f}%)")

    # 打印每个子集的类别分布
    for name, labels in [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]:
        print(f"\n{name}类别分布:")
        for i in range(config.NUM_CLASSES):
            count = np.sum(labels == i)
            percentage = count / len(labels) * 100 if len(labels) > 0 else 0
            print(f"  {config.IDX_TO_CLASS[i]:15s}: {count:5d} ({percentage:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_data(dataset_mode='mixed', single_dataset='UTD',
                normalize=True, normalization_method='standard'):
    """
    准备数据的完整流程

    Args:
        dataset_mode: 'single' 或 'mixed'
        single_dataset: 如果是single模式，使用哪个数据集
        normalize: 是否归一化
        normalization_method: 归一化方法

    Returns:
        字典，包含所有划分后的数据
    """
    print("\n" + "="*80)
    print("数据准备")
    print("="*80)
    print(f"模式: {dataset_mode}")

    # 加载数据
    if dataset_mode == 'single':
        print(f"数据集: {single_dataset}")
        data, labels, sample_info = load_single_dataset(single_dataset)
    elif dataset_mode == 'mixed':
        print(f"数据集: {', '.join(config.DATASETS)}")
        data, labels, sample_info = load_mixed_datasets()
    else:
        raise ValueError(f"未知的数据集模式: {dataset_mode}")

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data, labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )

    # 归一化
    scaler = None
    if normalize:
        print(f"\n应用{normalization_method}归一化...")
        X_train, X_val, X_test, scaler = normalize_data(
            X_train, X_val, X_test, method=normalization_method
        )
        print("归一化完成")

    print("="*80 + "\n")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'sample_info': sample_info
    }

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=64, num_workers=4):
    """
    创建PyTorch DataLoader

    Args:
        X_train, X_val, X_test: 数据
        y_train, y_val, y_test: 标签
        batch_size: 批次大小
        num_workers: 工作进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = GNSSDataset(X_train, y_train)
    val_dataset = GNSSDataset(X_val, y_val)
    test_dataset = GNSSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
