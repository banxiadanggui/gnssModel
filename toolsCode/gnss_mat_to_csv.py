#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
GNSS MAT to CSV Converter
=========================

功能：将GNSS接收机处理后的MAT文件转换为CSV格式
作者：GNSS Data Processing Toolbox
日期：2025-11-14

使用方法：
    python gnss_mat_to_csv.py <mat_file_path> [output_dir]

示例：
    python gnss_mat_to_csv.py D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat
    python gnss_mat_to_csv.py input.mat D:\output

输入参数：
    mat_file_path  - MAT文件的完整路径
    output_dir     - (可选) 输出目录，默认为MAT文件所在目录的csv_output子目录

输出：
    生成多个CSV文件，包含不同类型的数据：
    - *_trackData_SIGNAL.csv     - 跟踪数据（每个信号类型一个文件）
    - *_navData_Position.csv     - 导航位置数据
    - *_navData_Velocity.csv     - 导航速度数据
    - *_obsData_SIGNAL.csv       - 观测数据（每个信号类型一个文件）
    - *_statistics.csv           - 统计结果
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_mat_file(mat_file_path):
    """
    加载MAT文件，自动检测格式并使用合适的加载器

    Args:
        mat_file_path: MAT文件路径

    Returns:
        dict: 包含MAT文件数据的字典
    """
    print(f"\n正在加载MAT文件: {mat_file_path}")

    # 首先尝试使用scipy.io（支持MATLAB v7.2及以下）
    if HAS_SCIPY:
        try:
            # 不使用squeeze_me，避免结构体被展平为ndarray
            data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=False)
            print("[OK] 使用scipy.io成功加载MAT文件")
            return data
        except NotImplementedError:
            print("  scipy.io无法加载（可能是v7.3格式），尝试使用h5py...")
        except Exception as e:
            print(f"  scipy.io加载失败: {e}")

    # 尝试使用h5py（支持MATLAB v7.3 HDF5格式）
    if HAS_H5PY:
        try:
            data = {}
            with h5py.File(mat_file_path, 'r') as f:
                data = load_h5py_data(f)
            print("[OK] 使用h5py成功加载MAT文件")
            return data
        except Exception as e:
            print(f"  h5py加载失败: {e}")

    raise RuntimeError("无法加载MAT文件。请安装scipy或h5py库。\n"
                      "安装命令: pip install scipy h5py")


def load_h5py_data(h5_obj):
    """
    递归加载HDF5格式的MAT文件数据

    Args:
        h5_obj: h5py File或Group对象

    Returns:
        dict: 转换后的数据字典
    """
    data = {}

    for key in h5_obj.keys():
        if key.startswith('#'):  # 跳过元数据
            continue

        item = h5_obj[key]

        if isinstance(item, h5py.Dataset):
            # 数据集
            data[key] = np.array(item)
        elif isinstance(item, h5py.Group):
            # 组（结构体）
            data[key] = load_h5py_data(item)

    return data


def get_struct_field(struct, field_name, default=None):
    """
    安全地从结构体中获取字段值

    Args:
        struct: MATLAB结构体、数组或字典
        field_name: 字段名
        default: 默认值

    Returns:
        字段值或默认值
    """
    # 如果是numpy数组，尝试提取第一个元素
    if isinstance(struct, np.ndarray):
        if struct.size == 1:
            # 单元素数组，提取元素
            struct = struct.flat[0]
        elif struct.size > 0:
            # 多元素数组，提取第一个元素
            struct = struct.flat[0]
        else:
            return default

    # 处理字典
    if isinstance(struct, dict):
        return struct.get(field_name, default)

    # 处理MATLAB结构体对象
    elif hasattr(struct, field_name):
        field_value = getattr(struct, field_name)
        # 如果字段值也是被包装的单元素数组，提取它
        if isinstance(field_value, np.ndarray) and field_value.size == 1:
            return field_value.flat[0]
        return field_value
    else:
        return default


def ensure_list(obj):
    """确保对象是列表形式"""
    if isinstance(obj, np.ndarray):
        # 展平数组并转换为列表
        flat = obj.flatten()
        result = []
        for item in flat:
            # 处理字节字符串
            if isinstance(item, bytes):
                result.append(item.decode('utf-8'))
            elif isinstance(item, np.ndarray) and item.size == 1:
                # 嵌套的单元素数组
                result.append(item.item())
            else:
                result.append(item)
        return result
    elif isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def export_track_data(track_data, settings, output_dir, base_name):
    """
    导出跟踪数据到CSV

    Args:
        track_data: trackData结构
        settings: 配置参数
        output_dir: 输出目录
        base_name: 输出文件基础名称
    """
    print("\n导出跟踪数据...")

    enabled_signals = get_struct_field(
        get_struct_field(settings, 'sys', {}),
        'enabledSignals',
        []
    )
    enabled_signals = ensure_list(enabled_signals)

    for signal in enabled_signals:
        if isinstance(signal, bytes):
            signal = signal.decode('utf-8')
        signal = str(signal).strip()

        signal_data = get_struct_field(track_data, signal)
        if signal_data is None:
            print(f"  [WARNING] 信号 {signal} 未在trackData中找到")
            continue

        print(f"  处理信号: {signal}")

        channels = get_struct_field(signal_data, 'channel', [])
        if not isinstance(channels, (list, np.ndarray)):
            channels = [channels]

        all_channel_data = []
        global_epoch_offset = 0  # 全局Epoch偏移量

        for ch_idx, ch in enumerate(channels, start=1):
            # 获取提示相位I通道数据以确定长度
            i_p = get_struct_field(ch, 'I_P', [])
            if i_p is None or len(i_p) == 0:
                continue

            i_p = np.array(i_p).flatten()
            data_length = len(i_p)

            # 获取卫星ID
            sv_id_struct = get_struct_field(ch, 'SvId', {})
            sat_id = get_struct_field(sv_id_struct, 'satId', ch_idx)

            # 创建通道数据字典
            # 使用全局Epoch（每个通道接续上一个通道的Epoch）
            channel_dict = {
                'GlobalEpoch': np.arange(global_epoch_offset + 1, global_epoch_offset + data_length + 1),
                'LocalEpoch': np.arange(1, data_length + 1),  # 保留通道内的本地Epoch
                'ChannelID': np.full(data_length, ch_idx),
                'SatelliteID': np.full(data_length, sat_id),
                'SignalType': [signal] * data_length,
            }

            global_epoch_offset += data_length  # 更新全局偏移量

            # 时间序列字段映射（长度应该等于data_length）
            time_series_mapping = {
                # 原有字段
                'I_P': 'I_Prompt',
                'Q_P': 'Q_Prompt',
                'carrFreq': 'CarrierFreq_Hz',
                'pllLockIndicator': 'PLL_Lock',
                'fllLockIndicator': 'FLL_Lock',

                # 新增时间序列字段
                'CN0fromSNR': 'CNo_dBHz',
                'meanCN0fromSNR': 'MeanCNo_dBHz',
                'noiseCNOfromSNR': 'NoiseCNo_dBHz',
                'doppler': 'Doppler_Hz',
                'dllDiscr': 'DLL_Discriminator',
                'absoluteSample': 'AbsoluteSample',
            }

            # 标量字段映射（长度=1，需要重复填充）
            scalar_mapping = {
                'I_E': 'I_Early_Config',
                'Q_E': 'Q_Early_Config',
                'I_L': 'I_Late_Config',
                'Q_L': 'Q_Late_Config',
                'codeFreq': 'CodeFreq_Hz_Config',
                'codeError': 'CodeError_Config',
                'carrError': 'CarrierError_Config',
                'fllDiscr': 'FLL_Discriminator_Config',
                'pllDiscr': 'PLL_Discriminator_Config',
            }

            # 长度不匹配字段（需要填充或裁剪）
            length_mismatch_mapping = {
                'carrPhase': ('CarrierPhase_rad', 377000),
                'codePhase': ('CodePhase_chips', 377000),
            }

            # 处理时间序列字段
            for mat_field, csv_field in time_series_mapping.items():
                field_data = get_struct_field(ch, mat_field)
                if field_data is not None:
                    field_data = np.array(field_data).flatten()
                    if len(field_data) == data_length:
                        channel_dict[csv_field] = field_data
                    elif len(field_data) > 0:
                        # 长度不匹配，尝试填充或裁剪
                        if len(field_data) > data_length:
                            # 裁剪
                            channel_dict[csv_field] = field_data[:data_length]
                        else:
                            # 填充NaN
                            padded = np.full(data_length, np.nan)
                            padded[:len(field_data)] = field_data
                            channel_dict[csv_field] = padded

            # 处理标量字段（重复填充）
            for mat_field, csv_field in scalar_mapping.items():
                field_data = get_struct_field(ch, mat_field)
                if field_data is not None:
                    # 提取标量值
                    if isinstance(field_data, np.ndarray):
                        scalar_value = field_data.flatten()[0] if field_data.size > 0 else np.nan
                    else:
                        scalar_value = field_data
                    # 重复填充到所有行
                    channel_dict[csv_field] = np.full(data_length, scalar_value)

            # 处理长度不匹配字段（填充）
            for mat_field, (csv_field, expected_length) in length_mismatch_mapping.items():
                field_data = get_struct_field(ch, mat_field)
                if field_data is not None:
                    field_data = np.array(field_data).flatten()
                    if len(field_data) == expected_length:
                        # 填充NaN到data_length
                        padded = np.full(data_length, np.nan)
                        padded[:expected_length] = field_data
                        channel_dict[csv_field] = padded
                    elif len(field_data) == data_length:
                        # 长度刚好匹配
                        channel_dict[csv_field] = field_data

            all_channel_data.append(pd.DataFrame(channel_dict))

        # 合并所有通道数据
        if all_channel_data:
            df = pd.concat(all_channel_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"{base_name}_trackData_{signal}.csv")
            df.to_csv(output_file, index=False)
            print(f"    [OK] 已保存: {base_name}_trackData_{signal}.csv "
                  f"({len(df)} 行, {len(all_channel_data)} 通道)")


def export_nav_data(nav_data, output_dir, base_name):
    """
    导出导航数据到CSV

    Args:
        nav_data: navData结构
        output_dir: 输出目录
        base_name: 输出文件基础名称
    """
    print("\n导出导航数据...")

    # 导出位置数据
    pos_data = get_struct_field(nav_data, 'Pos')
    if pos_data is not None:
        pos_dict = {}

        # XYZ坐标
        xyz = get_struct_field(pos_data, 'xyz')
        if xyz is not None:
            xyz = np.array(xyz)
            if xyz.ndim == 1:
                xyz = xyz.reshape(-1, 3)
            num_epochs = xyz.shape[0]
            pos_dict['Epoch'] = np.arange(1, num_epochs + 1)
            pos_dict['X_m'] = xyz[:, 0]
            pos_dict['Y_m'] = xyz[:, 1]
            pos_dict['Z_m'] = xyz[:, 2]

        # LLA坐标
        lla = get_struct_field(pos_data, 'LLA')
        if lla is not None:
            lla = np.array(lla)
            if lla.ndim == 1:
                lla = lla.reshape(-1, 3)
            if 'Epoch' not in pos_dict:
                num_epochs = lla.shape[0]
                pos_dict['Epoch'] = np.arange(1, num_epochs + 1)
            pos_dict['Latitude_deg'] = lla[:, 0]
            pos_dict['Longitude_deg'] = lla[:, 1]
            pos_dict['Altitude_m'] = lla[:, 2]

        # 其他字段
        simple_fields = {
            'dt': 'ClockBias_s',
            'nrSats': 'NumSatellites',
            'fom': 'FOM',
        }

        for mat_field, csv_field in simple_fields.items():
            field_data = get_struct_field(pos_data, mat_field)
            if field_data is not None:
                pos_dict[csv_field] = np.array(field_data).flatten()

        # DOP值
        dop = get_struct_field(pos_data, 'dop')
        if dop is not None:
            dop = np.array(dop)
            if dop.ndim == 1:
                dop = dop.reshape(-1, 5)
            if dop.shape[1] >= 5:
                pos_dict['GDOP'] = dop[:, 0]
                pos_dict['PDOP'] = dop[:, 1]
                pos_dict['HDOP'] = dop[:, 2]
                pos_dict['VDOP'] = dop[:, 3]
                pos_dict['TDOP'] = dop[:, 4]

        if pos_dict:
            df = pd.DataFrame(pos_dict)
            output_file = os.path.join(output_dir, f"{base_name}_navData_Position.csv")
            df.to_csv(output_file, index=False)
            print(f"  [OK] 已保存: {base_name}_navData_Position.csv ({len(df)} epochs)")

    # 导出速度数据
    vel_data = get_struct_field(nav_data, 'Vel')
    if vel_data is not None:
        vel_xyz = get_struct_field(vel_data, 'xyz')
        if vel_xyz is not None:
            vel_xyz = np.array(vel_xyz)
            if vel_xyz.ndim == 1:
                vel_xyz = vel_xyz.reshape(-1, 3)

            vel_dict = {
                'Epoch': np.arange(1, vel_xyz.shape[0] + 1),
                'VelX_m_s': vel_xyz[:, 0],
                'VelY_m_s': vel_xyz[:, 1],
                'VelZ_m_s': vel_xyz[:, 2],
            }

            fom = get_struct_field(vel_data, 'fom')
            if fom is not None:
                vel_dict['VelFOM'] = np.array(fom).flatten()

            df = pd.DataFrame(vel_dict)
            output_file = os.path.join(output_dir, f"{base_name}_navData_Velocity.csv")
            df.to_csv(output_file, index=False)
            print(f"  [OK] 已保存: {base_name}_navData_Velocity.csv ({len(df)} epochs)")


def export_obs_data(obs_data, settings, output_dir, base_name):
    """
    导出观测数据到CSV

    Args:
        obs_data: obsData结构
        settings: 配置参数
        output_dir: 输出目录
        base_name: 输出文件基础名称
    """
    print("\n导出观测数据...")

    enabled_signals = get_struct_field(
        get_struct_field(settings, 'sys', {}),
        'enabledSignals',
        []
    )
    enabled_signals = ensure_list(enabled_signals)

    for signal in enabled_signals:
        if isinstance(signal, bytes):
            signal = signal.decode('utf-8')
        signal = str(signal).strip()

        signal_obs = get_struct_field(obs_data, signal)
        if signal_obs is None:
            print(f"  [WARNING] 信号 {signal} 未在obsData中找到")
            continue

        print(f"  处理信号: {signal}")

        channels = get_struct_field(signal_obs, 'channel', [])
        if not isinstance(channels, (list, np.ndarray)):
            channels = [channels]

        all_obs_data = []
        global_epoch_offset = 0  # 全局Epoch偏移量

        for ch_idx, ch in enumerate(channels, start=1):
            raw_p = get_struct_field(ch, 'rawP', [])
            if raw_p is None or len(raw_p) == 0:
                continue

            raw_p = np.array(raw_p).flatten()
            data_length = len(raw_p)

            # 获取卫星ID
            sv_id_struct = get_struct_field(ch, 'SvId', {})
            sat_id = get_struct_field(sv_id_struct, 'satId', ch_idx)

            obs_dict = {
                'GlobalEpoch': np.arange(global_epoch_offset + 1, global_epoch_offset + data_length + 1),
                'LocalEpoch': np.arange(1, data_length + 1),
                'ChannelID': np.full(data_length, ch_idx),
                'SatelliteID': np.full(data_length, sat_id),
            }

            global_epoch_offset += data_length  # 更新全局偏移量

            # 观测数据字段
            field_mapping = {
                'rawP': 'Pseudorange_m',
                'corrP': 'CorrectedPseudorange_m',
                'phase': 'CarrierPhase_cycles',
                'doppler': 'Doppler_Hz',
                'CNo': 'CNo_dBHz',
            }

            for mat_field, csv_field in field_mapping.items():
                field_data = get_struct_field(ch, mat_field)
                if field_data is not None:
                    field_data = np.array(field_data).flatten()
                    if len(field_data) == data_length:
                        obs_dict[csv_field] = field_data

            all_obs_data.append(pd.DataFrame(obs_dict))

        if all_obs_data:
            df = pd.concat(all_obs_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"{base_name}_obsData_{signal}.csv")
            df.to_csv(output_file, index=False)
            print(f"    [OK] 已保存: {base_name}_obsData_{signal}.csv ({len(df)} 行)")


def export_statistics(stat_results, output_dir, base_name):
    """
    导出统计结果到CSV

    Args:
        stat_results: statResults结构
        output_dir: 输出目录
        base_name: 输出文件基础名称
    """
    print("\n导出统计结果...")

    stat_dict = {}

    # 水平误差统计
    hor = get_struct_field(stat_results, 'hor')
    if hor is not None:
        stat_dict['Horizontal_Mean_m'] = get_struct_field(hor, 'mean')
        stat_dict['Horizontal_Std_m'] = get_struct_field(hor, 'std')
        stat_dict['Horizontal_RMS_m'] = get_struct_field(hor, 'rms')
        stat_dict['Horizontal_95th_m'] = get_struct_field(hor, 'p95')

    # 垂直误差统计
    ver = get_struct_field(stat_results, 'ver')
    if ver is not None:
        stat_dict['Vertical_Mean_m'] = get_struct_field(ver, 'mean')
        stat_dict['Vertical_Std_m'] = get_struct_field(ver, 'std')
        stat_dict['Vertical_RMS_m'] = get_struct_field(ver, 'rms')
        stat_dict['Vertical_95th_m'] = get_struct_field(ver, 'p95')

    # 3D RMS
    rms_3d = get_struct_field(stat_results, 'RMS3D')
    if rms_3d is not None:
        stat_dict['RMS_3D_m'] = rms_3d

    # DOP统计
    dop = get_struct_field(stat_results, 'dop')
    if dop is not None:
        stat_dict['Mean_GDOP'] = get_struct_field(dop, 'meanGDOP')
        stat_dict['Mean_PDOP'] = get_struct_field(dop, 'meanPDOP')
        stat_dict['Mean_HDOP'] = get_struct_field(dop, 'meanHDOP')
        stat_dict['Mean_VDOP'] = get_struct_field(dop, 'meanVDOP')
        stat_dict['Mean_TDOP'] = get_struct_field(dop, 'meanTDOP')

    # 移除None值
    stat_dict = {k: v for k, v in stat_dict.items() if v is not None}

    if stat_dict:
        df = pd.DataFrame([stat_dict])
        output_file = os.path.join(output_dir, f"{base_name}_statistics.csv")
        df.to_csv(output_file, index=False)
        print(f"  [OK] 已保存: {base_name}_statistics.csv")
    else:
        print("  [WARNING] 无统计数据可导出")


def convert_mat_to_csv(mat_file_path, output_dir=None):
    """
    将MAT文件转换为CSV文件

    Args:
        mat_file_path: MAT文件路径
        output_dir: 输出目录（可选）
    """
    # 检查文件是否存在
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT文件不存在: {mat_file_path}")

    # 设置输出目录
    if output_dir is None:
        input_dir = os.path.dirname(mat_file_path)
        base_name = Path(mat_file_path).stem
        output_dir = os.path.join(input_dir, 'csv_output', base_name)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取基础文件名
    base_name = Path(mat_file_path).stem

    print("\n" + "=" * 60)
    print("GNSS MAT to CSV Converter")
    print("=" * 60)
    print(f"输入文件:  {mat_file_path}")
    print(f"输出目录:  {output_dir}")
    print("=" * 60)

    # 加载MAT文件
    data = load_mat_file(mat_file_path)

    # 导出各类数据
    if 'trackData' in data:
        export_track_data(data['trackData'], data.get('settings', {}), output_dir, base_name)

    if 'navData' in data:
        export_nav_data(data['navData'], output_dir, base_name)

    if 'obsData' in data:
        export_obs_data(data['obsData'], data.get('settings', {}), output_dir, base_name)

    if 'statResults' in data:
        export_statistics(data['statResults'], output_dir, base_name)

    print("\n" + "=" * 60)
    print("转换成功完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")


def main():
    """主函数：解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description='将GNSS接收机处理后的MAT文件转换为CSV格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s D:\\skill\\beidou\\data\\processedMAT\\UTD_processed_latest.mat
  %(prog)s input.mat D:\\output
        """
    )

    parser.add_argument(
        'mat_file',
        help='MAT文件的完整路径'
    )

    parser.add_argument(
        'output_dir',
        nargs='?',
        default=None,
        help='输出目录（可选），默认为MAT文件所在目录的csv_output子目录'
    )

    args = parser.parse_args()

    try:
        convert_mat_to_csv(args.mat_file, args.output_dir)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
