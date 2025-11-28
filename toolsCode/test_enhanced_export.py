#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试增强版导出功能"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from gnss_mat_to_csv import load_mat_file, export_track_data

# 测试参数
mat_file = r'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat'
output_dir = r'D:\skill\beidou\data\processedMAT\csv_output\test_enhanced'
base_name = 'test_enhanced'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("测试增强版MAT到CSV转换")
print("=" * 70)

# 加载MAT文件
print("\n正在加载MAT文件...")
data = load_mat_file(mat_file)

# 导出trackData（只处理GPS L1）
if 'trackData' in data and 'settings' in data:
    print("\n开始导出trackData...")

    try:
        export_track_data(data['trackData'], data['settings'], output_dir, base_name)

        print("\n" + "=" * 70)
        print("导出完成！")
        print("=" * 70)

        # 检查生成的CSV
        import pandas as pd
        import glob

        csv_files = glob.glob(os.path.join(output_dir, '*.csv'))

        for csv_file in csv_files:
            print(f"\n检查文件: {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file, nrows=10)

            print(f"  行数: {len(df)} (仅读取前10行测试)")
            print(f"  列数: {len(df.columns)}")
            print(f"  列名: {list(df.columns)}")

            # 显示前3行数据
            print("\n  前3行数据预览:")
            print(df.head(3).to_string(max_colwidth=15))

            # 检查新增字段
            new_fields = ['CNo_dBHz', 'Doppler_Hz', 'CarrierPhase_rad',
                         'CodePhase_chips', 'I_Early_Config']
            found_new = [f for f in new_fields if f in df.columns]
            print(f"\n  新增字段 ({len(found_new)}/{len(new_fields)}): {found_new}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("错误: MAT文件中没有找到trackData或settings")
