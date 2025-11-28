#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查Excel文件内容"""

import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

excel_file = r'D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\UTD_processed_latest_trackData_gpsl1.xlsx'

print(f"检查Excel文件: {excel_file}\n")
print("=" * 70)

# 读取Excel
df = pd.read_excel(excel_file, nrows=20)

print("前20行数据预览:")
print("=" * 70)
print(df.to_string())

print("\n" + "=" * 70)
print("数据类型:")
print("=" * 70)
print(df.dtypes)

print("\n" + "=" * 70)
print("示例数值（原始格式）:")
print("=" * 70)
for col in ['I_Prompt', 'Q_Prompt', 'CarrierFreq_Hz', 'PLL_Lock', 'FLL_Lock']:
    if col in df.columns:
        val = df[col].iloc[0]
        print(f"{col}: {val} (type: {type(val).__name__})")
