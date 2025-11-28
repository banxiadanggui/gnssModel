"""
专门用于转换 UTD_processed_latest.mat 文件的脚本
将复杂的嵌套结构体分别导出为不同的 CSV 文件
"""
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

def extract_struct_fields(struct_data, prefix=''):
    """提取结构体中的所有字段"""
    result = {}

    if struct_data.dtype.names is None:
        return {prefix: struct_data}

    for field_name in struct_data.dtype.names:
        field_data = struct_data[field_name]
        full_name = f"{prefix}_{field_name}" if prefix else field_name

        # 提取数据（处理 (1,1) 包装）
        if isinstance(field_data, np.ndarray):
            if field_data.shape == (1, 1):
                field_data = field_data[0, 0]
            elif field_data.shape[0] == 1:
                field_data = field_data[0]

        # 如果还是结构体，递归处理
        if isinstance(field_data, np.ndarray) and field_data.dtype.names is not None:
            result.update(extract_struct_fields(field_data, full_name))
        else:
            result[full_name] = field_data

    return result

def convert_struct_to_csv(struct_data, output_file, struct_name=''):
    """将结构体转换为CSV"""
    print(f"\n正在处理 {struct_name}...")

    fields = extract_struct_fields(struct_data)

    # 构建 DataFrame
    df_dict = {}
    max_len = 0

    for field_name, field_data in fields.items():
        if isinstance(field_data, np.ndarray):
            # 处理多维数组
            if field_data.ndim == 1:
                df_dict[field_name] = field_data
                max_len = max(max_len, len(field_data))
            elif field_data.ndim == 2:
                # 2D数组，每列作为一个字段
                if field_data.shape[0] == 1:
                    # 如果第一维是1，转置
                    field_data = field_data.T

                for i in range(field_data.shape[1]):
                    df_dict[f"{field_name}_col{i+1}"] = field_data[:, i]
                max_len = max(max_len, field_data.shape[0])
            else:
                # 更高维度，展平
                df_dict[field_name] = field_data.flatten()
                max_len = max(max_len, len(field_data.flatten()))
        elif isinstance(field_data, (int, float)):
            df_dict[field_name] = field_data

    if not df_dict:
        print(f"  警告: 未找到数据")
        return False

    # 对齐所有列的长度
    aligned_dict = {}
    for key, value in df_dict.items():
        if isinstance(value, np.ndarray):
            # 检查是否包含嵌套数组
            if value.dtype == object or str(value.dtype).startswith('object'):
                print(f"  跳过复杂字段: {key} (包含嵌套结构)")
                continue

            if len(value) == max_len:
                aligned_dict[key] = value
            elif len(value) == 1:
                # 标量，复制到所有行
                try:
                    aligned_dict[key] = np.full(max_len, value[0])
                except:
                    print(f"  跳过字段: {key} (无法转换)")
                    continue
            else:
                # 长度不匹配，填充NaN
                try:
                    padded = np.full(max_len, np.nan)
                    padded[:len(value)] = value
                    aligned_dict[key] = padded
                except:
                    print(f"  跳过字段: {key} (长度不匹配且无法填充)")
                    continue
        else:
            # 标量值
            try:
                aligned_dict[key] = np.full(max_len, value)
            except:
                print(f"  跳过字段: {key} (无法转换标量)")

    if aligned_dict:
        df = pd.DataFrame(aligned_dict)
        df.to_csv(output_file, index=False, float_format='%.10e')
        print(f"  成功: {output_file}")
        print(f"  形状: {df.shape}")
        return True

    return False

def convert_utd_mat(mat_file, output_dir):
    """转换 UTD MAT 文件"""
    print(f"读取文件: {mat_file}")

    # 读取MAT文件
    mat_data = loadmat(mat_file)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 主要字段
    main_fields = ['settings', 'acqData', 'trackData', 'ephData', 'statResults']

    success_count = 0
    for field_name in main_fields:
        if field_name in mat_data:
            output_file = os.path.join(output_dir, f"UTD_{field_name}.csv")

            # 提取字段（通常是 (1,1) 结构体）
            field_data = mat_data[field_name]
            if field_data.shape == (1, 1):
                field_data = field_data[0, 0]

            if convert_struct_to_csv(field_data, output_file, field_name):
                success_count += 1

    # 处理 obsData, satData, navData（时间序列数据）
    time_series_fields = ['obsData', 'satData', 'navData']
    for field_name in time_series_fields:
        if field_name in mat_data:
            print(f"\n正在处理 {field_name} (时间序列)...")
            field_data = mat_data[field_name]
            print(f"  形状: {field_data.shape}")
            print(f"  这是一个时间序列数据，每个元素是一个时间点")
            print(f"  提示: 这种数据需要特殊处理，暂不导出")

    print(f"\n\n转换完成! 成功导出 {success_count} 个文件")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    mat_file = r"D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat"
    output_dir = r"D:\skill\beidou\data\processedCSV"

    convert_utd_mat(mat_file, output_dir)
