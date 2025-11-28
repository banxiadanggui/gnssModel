#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV诊断工具 - 检查空行问题
"""

import pandas as pd
import sys

def diagnose_csv(csv_file):
    """诊断CSV文件中的空行问题"""

    print(f"\n正在分析: {csv_file}\n")
    print("=" * 60)

    # 读取CSV
    df = pd.read_csv(csv_file)

    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}\n")

    # 检查完全空行
    empty_rows = df.isnull().all(axis=1).sum()
    print(f"完全空行数: {empty_rows}")

    # 检查部分空值
    partial_empty = df.isnull().any(axis=1).sum()
    print(f"包含空值的行数: {partial_empty}\n")

    # 检查Epoch重复
    if 'Epoch' in df.columns:
        epoch_counts = df['Epoch'].value_counts()
        max_repeat = epoch_counts.max()
        print(f"Epoch值范围: {df['Epoch'].min()} - {df['Epoch'].max()}")
        print(f"Epoch最大重复次数: {max_repeat}")

        if max_repeat > 1:
            print(f"\n⚠ 发现Epoch重复! 这可能导致CSV显示异常")
            print(f"   原因: 每个卫星通道的Epoch都从1开始")

            # 显示重复的Epoch样本
            repeated = epoch_counts[epoch_counts > 1].head(5)
            print(f"\n重复Epoch示例:")
            for epoch, count in repeated.items():
                print(f"   Epoch {epoch}: 出现 {count} 次")

    # 检查卫星通道分布
    if 'SatelliteID' in df.columns:
        sat_counts = df['SatelliteID'].value_counts().sort_index()
        print(f"\n卫星通道分布:")
        for sat_id, count in sat_counts.items():
            print(f"   卫星 {sat_id}: {count} 行")

    # 检查每列的空值情况
    print(f"\n各列空值统计:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"   {col}: {count} ({percentage:.1f}%)")

    # 显示前几行数据
    print(f"\n前10行数据预览:")
    print(df.head(10).to_string())

    print("\n" + "=" * 60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose_csv.py <csv_file>")
        print("示例: python diagnose_csv.py output.csv")
        sys.exit(1)

    diagnose_csv(sys.argv[1])
