#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比MAT文件和Excel中的字段"""

import scipy.io as sio
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 文件路径
mat_file = r'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat'
csv_file = r'D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\UTD_processed_latest_trackData_gpsl1.csv'

print("=" * 80)
print("MAT文件 vs Excel字段对比分析")
print("=" * 80)

# 加载MAT文件
print("\n[1] 加载MAT文件...")
data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=False)
trackData = data['trackData']

if hasattr(trackData, 'flatten'):
    trackData = trackData.flatten()[0]

# 获取GPS L1信号数据
gpsl1 = getattr(trackData, 'gpsl1')
if hasattr(gpsl1, 'flatten'):
    gpsl1 = gpsl1.flatten()[0]

# 获取第一个通道
channels = gpsl1.channel
if hasattr(channels, 'flatten'):
    channels = channels.flatten()

first_channel = channels[0]

print(f"信号类型: gpsl1")
print(f"通道数: {len(channels)}")
print(f"第一个通道 (卫星ID {first_channel.SvId.flatten()[0].satId})")

# 获取所有字段及其shape
print("\n[2] MAT文件中的所有字段:")
print("-" * 80)

all_fields = {}
if hasattr(first_channel, '_fieldnames'):
    field_names = first_channel._fieldnames

    for field_name in sorted(field_names):
        field_data = getattr(first_channel, field_name)

        if hasattr(field_data, 'shape'):
            shape = field_data.shape
            dtype = field_data.dtype
            all_fields[field_name] = {
                'shape': shape,
                'dtype': dtype,
                'length': shape[1] if len(shape) > 1 else shape[0]
            }
        else:
            all_fields[field_name] = {
                'shape': 'N/A',
                'dtype': type(field_data).__name__,
                'length': 'N/A'
            }

# 按长度分组
fields_by_length = {}
for field_name, info in all_fields.items():
    length = info['length']
    if length not in fields_by_length:
        fields_by_length[length] = []
    fields_by_length[length].append(field_name)

# 显示字段统计
print(f"\n总字段数: {len(all_fields)}")
print(f"\n按数据长度分组:")
for length in sorted(fields_by_length.keys(), key=lambda x: (x != 377900, x if isinstance(x, int) else 999999)):
    fields = fields_by_length[length]
    count = len(fields)
    if length == 377900:
        print(f"\n  [时间序列] 长度={length}: {count}个字段 ⭐ (应导出)")
    elif length == 377000:
        print(f"\n  [准时间序列] 长度={length}: {count}个字段 ⚠ (长度不匹配)")
    elif length == 1:
        print(f"\n  [标量/配置] 长度={length}: {count}个字段 ❌ (未导出)")
    else:
        print(f"\n  [其他] 长度={length}: {count}个字段")

    # 显示字段名（每行10个）
    for i in range(0, len(fields), 10):
        print(f"    {', '.join(fields[i:i+10])}")

# 加载CSV查看实际导出的字段
print("\n" + "=" * 80)
print("[3] Excel中实际导出的字段:")
print("-" * 80)

df = pd.read_csv(csv_file, nrows=1)
excel_cols = list(df.columns)

print(f"\n总列数: {len(excel_cols)}")
print(f"列名: {excel_cols}")

# 映射关系（脚本中的字段映射）
field_mapping = {
    'I_P': 'I_Prompt',
    'Q_P': 'Q_Prompt',
    'I_E': 'I_Early',
    'Q_E': 'Q_Early',
    'I_L': 'I_Late',
    'Q_L': 'Q_Late',
    'carrierPhase': 'CarrierPhase_rad',
    'carrFreq': 'CarrierFreq_Hz',
    'carrFreqRate': 'CarrierFreqRate_Hz_s',
    'codePhase': 'CodePhase_chips',
    'codeFreq': 'CodeFreq_Hz',
    'CNo': 'CNo_dBHz',
    'CN0fromSNR': 'CNo_dBHz',
    'pllLockIndicator': 'PLL_Lock',
    'fllLockIndicator': 'FLL_Lock',
    'dllLockIndicator': 'DLL_Lock',
}

# 找出MAT中有但Excel中缺失的字段
print("\n" + "=" * 80)
print("[4] 详细对比 - 应导出但缺失的字段:")
print("-" * 80)

# 长度为377900的字段（时间序列）
time_series_fields = fields_by_length.get(377900, [])
print(f"\n长度=377900的时间序列字段 ({len(time_series_fields)}个):")

exported_mat_fields = []
missing_mat_fields = []

for mat_field in time_series_fields:
    csv_field = field_mapping.get(mat_field, None)

    if csv_field and csv_field in excel_cols:
        exported_mat_fields.append((mat_field, csv_field, '✓'))
    else:
        # 检查是否有其他匹配
        found = False
        for excel_col in excel_cols:
            if mat_field.lower() in excel_col.lower() or excel_col.lower() in mat_field.lower():
                exported_mat_fields.append((mat_field, excel_col, '✓'))
                found = True
                break

        if not found:
            missing_mat_fields.append(mat_field)

print(f"\n已导出的字段 ({len(exported_mat_fields)}个):")
for mat_field, csv_field, status in sorted(exported_mat_fields):
    info = all_fields[mat_field]
    print(f"  {status} {mat_field:<30} -> {csv_field:<30} shape={info['shape']}")

print(f"\n缺失的字段 ({len(missing_mat_fields)}个):")
for mat_field in sorted(missing_mat_fields):
    info = all_fields[mat_field]
    shape_str = str(info['shape'])
    print(f"  [X] {mat_field:<30} shape={shape_str:<20} dtype={info['dtype']}")

# 长度不匹配的字段
print("\n" + "=" * 80)
print("[5] 长度不匹配的字段（无法直接导出）:")
print("-" * 80)

for length, fields in sorted(fields_by_length.items()):
    if length not in [377900, 1, 'N/A'] and isinstance(length, int):
        print(f"\n长度={length} ({len(fields)}个字段):")
        for field in sorted(fields):
            info = all_fields[field]
            shape_str = str(info['shape'])
            # 检查是否在映射表中
            if field in field_mapping:
                print(f"  [!] {field:<30} -> {field_mapping[field]:<30} shape={shape_str}")
            else:
                print(f"    {field:<30} shape={shape_str}")

# 标量字段（配置参数）
print("\n" + "=" * 80)
print("[6] 标量/配置字段（长度=1，未导出）:")
print("-" * 80)

scalar_fields = fields_by_length.get(1, [])
print(f"\n总共 {len(scalar_fields)} 个标量字段")
print("这些字段每个卫星只有一个值（配置参数、最终状态等）\n")

# 按类别分组显示
categories = {
    '相关器配置': ['I_E', 'Q_E', 'I_L', 'Q_L', 'I_E_E', 'Q_E_E', 'corrFingers',
                   'earlyFingerIndex', 'promptFingerIndex', 'lateFingerIndex', 'noiseFingerIndex'],
    '环路参数': ['codeFreq', 'codeError', 'codeNco', 'carrError', 'carrFreq',
                'tau1carr', 'tau2carr', 'tau1code', 'tau2code'],
    '锁定阈值': ['pllWideBandLockIndicatorThreshold', 'pllNarrowBandLockIndicatorThreshold',
                'fllWideBandLockIndicatorThreshold', 'fllNarrowBandLockIndicatorThreshold'],
    '噪声带宽': ['fllNoiseBandwidthWide', 'fllNoiseBandwidthNarrow', 'fllNoiseBandwidthVeryNarrow',
                'pllNoiseBandwidthWide', 'pllNoiseBandwidthNarrow', 'pllNoiseBandwidthVeryNarrow'],
    '鉴别器': ['fllDiscr', 'pllDiscr'],
    '滤波器': ['fllFilter', 'pllFilter', 'fllLoopGain', 'pllLoopGain'],
    '其他配置': ['bitSync', 'bitValue', 'carrierFreq', 'intermediateFreq'],
}

for category, expected_fields in categories.items():
    found = [f for f in expected_fields if f in scalar_fields]
    if found:
        print(f"{category} ({len(found)}个):")
        for field in found:
            info = all_fields[field]
            mapped = field_mapping.get(field, '(无映射)')
            print(f"  {field:<35} -> {mapped:<30} dtype={info['dtype']}")
        print()

# 汇总统计
print("=" * 80)
print("[7] 汇总统计:")
print("-" * 80)

print(f"""
MAT文件中的字段:
  - 总字段数: {len(all_fields)}
  - 时间序列字段 (长度=377900): {len(time_series_fields)}
  - 准时间序列 (长度=377000): {len(fields_by_length.get(377000, []))}
  - 标量字段 (长度=1): {len(scalar_fields)}
  - 其他长度: {sum(len(v) for k, v in fields_by_length.items() if k not in [377900, 377000, 1, 'N/A'])}

Excel中导出的列:
  - 总列数: {len(excel_cols)}
  - 元数据列: 5 (GlobalEpoch, LocalEpoch, ChannelID, SatelliteID, SignalType)
  - 数据列: {len(excel_cols) - 5}

导出情况:
  - 已导出的时间序列字段: {len(exported_mat_fields)} / {len(time_series_fields)}
  - 缺失的时间序列字段: {len(missing_mat_fields)}
  - 未导出的标量字段: {len(scalar_fields)}
  - 未导出的准时间序列: {len(fields_by_length.get(377000, []))}
""")

print("=" * 80)
print("建议:")
print("=" * 80)
print("""
1. 缺失的时间序列字段可以添加到field_mapping并导出
2. 标量字段（长度=1）需要特殊处理：
   - 可以作为每行的重复值
   - 或导出为单独的配置文件
3. 长度不匹配的字段需要对齐或插值处理
""")
