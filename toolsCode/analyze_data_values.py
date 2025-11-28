#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析CSV数据值分布"""

import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

csv_file = r'D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\UTD_processed_latest_trackData_gpsl1.csv'

print(f"分析CSV文件: {csv_file}\n")

# 读取前1000行
df = pd.read_csv(csv_file, nrows=1000)

print("=" * 70)
print("数据值统计（前1000行）:")
print("=" * 70)

data_cols = ['I_Prompt', 'Q_Prompt', 'CarrierFreq_Hz', 'PLL_Lock', 'FLL_Lock']

for col in data_cols:
    if col in df.columns:
        vals = df[col].values
        print(f"\n{col}:")
        print(f"  范围: {vals.min():.4e} 到 {vals.max():.4e}")
        print(f"  平均: {vals.mean():.4e}")
        print(f"  接近零(|x|<1): {(np.abs(vals) < 1).sum()} / {len(vals)} ({(np.abs(vals) < 1).sum() / len(vals) * 100:.1f}%)")
        print(f"  非常小(|x|<0.01): {(np.abs(vals) < 0.01).sum()} / {len(vals)} ({(np.abs(vals) < 0.01).sum() / len(vals) * 100:.1f}%)")

# 检查每4行的模式
print("\n" + "=" * 70)
print("检查每4行的数据模式（前40行）:")
print("=" * 70)

for i in range(0, min(40, len(df)), 4):
    chunk = df.iloc[i:i+4]
    i_prompt_vals = chunk['I_Prompt'].values
    print(f"\n行 {i+1}-{i+4} (卫星 {chunk['SatelliteID'].iloc[0]}):")
    print(f"  I_Prompt: {i_prompt_vals}")
    print(f"  平均绝对值: {np.abs(i_prompt_vals).mean():.2f}")

# 检查是否有规律的低值模式
print("\n" + "=" * 70)
print("检查低值行的分布:")
print("=" * 70)

low_value_rows = (np.abs(df['I_Prompt']) < 100)
print(f"I_Prompt绝对值 < 100 的行数: {low_value_rows.sum()} / {len(df)}")

if low_value_rows.sum() > 0:
    low_indices = np.where(low_value_rows)[0]
    print(f"低值行的索引（前20个）: {low_indices[:20]}")

    # 检查是否有模式
    if len(low_indices) > 1:
        diffs = np.diff(low_indices)
        print(f"低值行之间的间隔（前20个）: {diffs[:20]}")
