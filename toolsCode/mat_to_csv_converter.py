import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.io import loadmat

def flatten_struct(data, parent_key=''):
    """将嵌套的MATLAB结构体展平为字典"""
    items = {}

    if isinstance(data, dict):
        for key, value in data.items():
            # 跳过MATLAB元数据
            if key.startswith('__'):
                continue
            new_key = f"{parent_key}_{key}" if parent_key else key

            if isinstance(value, np.ndarray):
                # 检查是否是结构体数组
                if value.dtype.names is not None:
                    # 结构体数组
                    for field in value.dtype.names:
                        # 安全地提取字段数据
                        try:
                            if value.ndim == 0:
                                field_data = value[field]
                            elif value.ndim == 1:
                                field_data = value[field][0]
                            else:
                                field_data = value[field][0, 0]
                            items.update(flatten_struct({field: field_data}, new_key))
                        except:
                            field_data = value[field]
                            if isinstance(field_data, np.ndarray):
                                items[f"{new_key}_{field}"] = field_data.flatten()
                            else:
                                items[f"{new_key}_{field}"] = field_data
                elif value.dtype == object or str(value.dtype).startswith('object'):
                    # 对象数组，可能包含嵌套结构
                    try:
                        if value.size == 1:
                            inner = value.flat[0]
                            if isinstance(inner, np.ndarray):
                                items[new_key] = inner.flatten() if inner.ndim > 1 else inner
                            elif isinstance(inner, dict):
                                items.update(flatten_struct(inner, new_key))
                            else:
                                items[new_key] = inner
                        else:
                            items[new_key] = value.flatten()
                    except:
                        items[new_key] = value.flatten()
                else:
                    # 普通数值数组
                    items[new_key] = value.flatten() if value.ndim > 1 else value
            elif isinstance(value, dict):
                items.update(flatten_struct(value, new_key))
            else:
                items[new_key] = value

    return items

def read_h5_dataset(dataset):
    """读取HDF5数据集，处理对象引用"""
    if dataset.dtype == 'object':
        # 对象类型，可能包含引用
        return None
    else:
        return dataset[...]

def extract_data_recursive(h5_group, prefix=''):
    """递归提取HDF5组中的所有数据"""
    data_dict = {}

    for key in h5_group.keys():
        full_key = f"{prefix}{key}" if prefix else key
        item = h5_group[key]

        if isinstance(item, h5py.Dataset):
            data = read_h5_dataset(item)
            if data is not None:
                # 如果是多维数组，展平它
                if len(data.shape) > 1:
                    # 对于2D数组，尝试按列保存
                    if len(data.shape) == 2:
                        for i in range(data.shape[1]):
                            data_dict[f"{full_key}_col{i}"] = data[:, i]
                    else:
                        data_dict[full_key] = data.flatten()
                else:
                    data_dict[full_key] = data
        elif isinstance(item, h5py.Group):
            # 递归处理组
            sub_data = extract_data_recursive(item, f"{full_key}_")
            data_dict.update(sub_data)

    return data_dict

def convert_mat_to_csv(mat_file_path, output_dir):
    """将MAT文件转换为CSV文件"""
    # 创建输出文件名
    mat_filename = Path(mat_file_path).stem
    output_file = os.path.join(output_dir, f"{mat_filename}.csv")

    print(f"正在处理: {mat_file_path}")

    # 首先尝试使用scipy读取（适用于旧版本MAT文件）
    try:
        print("  尝试使用scipy读取...")
        mat_data = loadmat(mat_file_path)
        all_data = flatten_struct(mat_data)

        if all_data:
            # 找到最大长度
            max_length = 0
            for key, value in all_data.items():
                if isinstance(value, np.ndarray):
                    length = len(value) if value.ndim == 1 else value.shape[0]
                    max_length = max(max_length, length)

            # 创建DataFrame
            df_dict = {}
            for key, value in all_data.items():
                if isinstance(value, np.ndarray):
                    length = len(value) if value.ndim == 1 else value.shape[0]
                    if length == max_length:
                        df_dict[key] = value
                    elif length == 1:
                        df_dict[key] = [value[0]] * max_length
                    else:
                        padded = np.full(max_length, np.nan)
                        padded[:length] = value
                        df_dict[key] = padded

            if df_dict:
                df = pd.DataFrame(df_dict)
                df.to_csv(output_file, index=False, float_format='%.10e')
                print(f"  成功保存到: {output_file}")
                print(f"  数据形状: {df.shape}")
                return True

        print(f"  警告: 未找到可提取的数据")
        return False

    except Exception as e:
        # 如果scipy失败，尝试使用h5py（适用于MATLAB v7.3+）
        print(f"  scipy读取失败: {str(e)}")
        print("  尝试使用h5py读取...")

    try:
        with h5py.File(mat_file_path, 'r') as f:
            # 提取所有数据
            all_data = {}

            # 遍历顶级键（忽略元数据）
            for key in f.keys():
                if not key.startswith('#'):
                    print(f"  提取 {key}...")
                    item = f[key]

                    if isinstance(item, h5py.Dataset):
                        data = read_h5_dataset(item)
                        if data is not None:
                            all_data[key] = data
                    elif isinstance(item, h5py.Group):
                        sub_data = extract_data_recursive(item, f"{key}_")
                        all_data.update(sub_data)

            # 找到最大长度
            max_length = 0
            for key, value in all_data.items():
                if isinstance(value, np.ndarray):
                    length = len(value) if value.ndim == 1 else value.shape[0]
                    max_length = max(max_length, length)

            # 创建DataFrame，对长度不同的数据进行填充
            df_dict = {}
            for key, value in all_data.items():
                if isinstance(value, np.ndarray):
                    length = len(value) if value.ndim == 1 else value.shape[0]
                    if length == max_length:
                        df_dict[key] = value
                    elif length == 1:
                        # 标量值，复制到所有行
                        df_dict[key] = [value[0]] * max_length
                    else:
                        # 长度不匹配，用NaN填充
                        padded = np.full(max_length, np.nan)
                        padded[:length] = value
                        df_dict[key] = padded

            if df_dict:
                df = pd.DataFrame(df_dict)
                df.to_csv(output_file, index=False, float_format='%.10e')
                print(f"  成功保存到: {output_file}")
                print(f"  数据形状: {df.shape}")
                return True
            else:
                print(f"  警告: 未找到可提取的数据")
                return False

    except Exception as e:
        print(f"  错误: {str(e)}")
        return False

def main():
    # 设置输入和输出路径
    data_dir = r"芬兰L1_E1数据集\芬兰L1_E1数据集"
    output_dir = r"processedData"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有mat文件
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    print(f"找到 {len(mat_files)} 个MAT文件\n")

    # 转换每个文件
    success_count = 0
    for mat_file in mat_files:
        mat_path = os.path.join(data_dir, mat_file)
        if convert_mat_to_csv(mat_path, output_dir):
            success_count += 1
        print()

    print(f"转换完成! 成功: {success_count}/{len(mat_files)}")

if __name__ == "__main__":
    main()
