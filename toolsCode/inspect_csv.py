#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查CSV文件数据质量"""

import pandas as pd
import sys
import io

# 设置输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取CSV文件
csv_file = r'D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\UTD_processed_latest_trackData_gpsl1.csv'

print(f"检查CSV文件: {csv_file}\n")
print("=" * 70)

# 读取前100行
df = pd.read_csv(csv_file, nrows=100)

print(f"前100行数据基本信息:")
print(f"  总行数: {len(df)}")
print(f"  总列数: {len(df.columns)}")
print(f"  列名: {list(df.columns)}")

print("\n" + "=" * 70)
print("空值统计:")
print("=" * 70)

# 检查每列的空值
for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    if null_count > 0:
        print(f"  {col:<25} {null_count:>3} / {len(df)} ({null_pct:.1f}%)")

# 统计完全为空的行
empty_rows = df.isnull().all(axis=1).sum()
print(f"\n完全为空的行: {empty_rows}")

# 统计部分为空的行
partial_empty = df.isnull().any(axis=1).sum()
print(f"包含空值的行: {partial_empty}")

print("\n" + "=" * 70)
print("前20行数据预览:")
print("=" * 70)
print(df.head(20).to_string())

print("\n" + "=" * 70)
print("数据类型:")
print("=" * 70)
print(df.dtypes)

# 检查是否有异常的行模式（每4行一个周期）
print("\n" + "=" * 70)
print("检查行模式（每4行）:")
print("=" * 70)

for i in range(0, min(20, len(df)), 4):
    chunk = df.iloc[i:i+4]
    null_counts = chunk.isnull().sum(axis=1)
    print(f"行 {i+1}-{i+4}: 空值数 = {null_counts.values}")
