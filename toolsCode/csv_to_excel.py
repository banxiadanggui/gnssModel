#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CSV to Excel Converter
======================

将CSV文件转换为Excel格式，解决WPS/VSCode显示问题

使用方法：
    python csv_to_excel.py <csv_file_or_folder>

示例：
    # 转换单个CSV文件
    python csv_to_excel.py data.csv

    # 转换整个文件夹中的所有CSV
    python csv_to_excel.py D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest
"""

import pandas as pd
import os
import sys
import glob

def csv_to_excel(csv_file, output_file=None, use_scientific_notation=True):
    """
    将CSV文件转换为Excel格式

    Args:
        csv_file: CSV文件路径
        output_file: 输出Excel文件路径（可选）
        use_scientific_notation: 是否将数值转换为科学计数法文本（可选）
    """
    if output_file is None:
        output_file = csv_file.rsplit('.', 1)[0] + '.xlsx'

    print(f"转换: {os.path.basename(csv_file)}")

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)

        # 将数值列转换为科学计数法文本
        if use_scientific_notation:
            for col in df.columns:
                # 跳过非数值列
                if col in ['SignalType', 'ChannelID', 'SatelliteID',
                          'GlobalEpoch', 'LocalEpoch']:
                    continue

                # 检查是否为数值类型
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 转换为科学计数法字符串
                    df[col] = df[col].apply(lambda x: f'{x:.10e}' if pd.notna(x) else '')

        # 创建Excel写入器，使用xlsxwriter引擎获得更多格式控制
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')

            # 获取工作簿和工作表对象
            workbook = writer.book
            worksheet = writer.sheets['Data']

            # 定义文本格式（左对齐）
            text_format = workbook.add_format({'align': 'left'})

            # 自动调整列宽并设置格式
            for idx, col in enumerate(df.columns):
                # 计算列宽（标题或最大值的长度）
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                # 设置列宽，最小15（科学计数法需要更宽），最大50
                width = min(max(max_len + 2, 15), 50)

                # 所有列都设置为文本格式
                worksheet.set_column(idx, idx, width, text_format)

            # 冻结首行（标题行）
            worksheet.freeze_panes(1, 0)

        print(f"  [OK] 已保存: {os.path.basename(output_file)}")
        print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
        return True

    except Exception as e:
        print(f"  [ERROR] 转换失败: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python csv_to_excel.py <csv_file_or_folder>")
        print("\n示例:")
        print("  python csv_to_excel.py data.csv")
        print("  python csv_to_excel.py D:\\data\\csv_output")
        sys.exit(1)

    path = sys.argv[1]

    print("\n" + "=" * 70)
    print("CSV to Excel Converter")
    print("=" * 70)

    if os.path.isfile(path):
        # 单个文件
        if not path.lower().endswith('.csv'):
            print(f"错误: {path} 不是CSV文件")
            sys.exit(1)

        csv_to_excel(path)

    elif os.path.isdir(path):
        # 文件夹中的所有CSV
        csv_files = glob.glob(os.path.join(path, '*.csv'))

        if not csv_files:
            print(f"错误: 在 {path} 中未找到CSV文件")
            sys.exit(1)

        print(f"\n找到 {len(csv_files)} 个CSV文件\n")

        success_count = 0
        for csv_file in csv_files:
            if csv_to_excel(csv_file):
                success_count += 1
            print()

        print("=" * 70)
        print(f"转换完成: {success_count}/{len(csv_files)} 成功")
        print("=" * 70)

    else:
        print(f"错误: {path} 不存在")
        sys.exit(1)


if __name__ == '__main__':
    main()
