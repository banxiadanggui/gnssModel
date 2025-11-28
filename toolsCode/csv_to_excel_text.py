#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CSV to Excel Converter (Pure Text Mode)
========================================

将CSV文件转换为Excel格式，所有数值列都转换为科学计数法文本

使用方法：
    python csv_to_excel_text.py <csv_file_or_folder>
"""

import pandas as pd
import xlsxwriter
import os
import sys
import glob

def csv_to_excel_text(csv_file, output_file=None):
    """
    将CSV转换为Excel，数值列转为科学计数法文本

    Args:
        csv_file: CSV文件路径
        output_file: 输出Excel文件路径
    """
    if output_file is None:
        output_file = csv_file.rsplit('.', 1)[0] + '_text.xlsx'

    print(f"转换: {os.path.basename(csv_file)}")

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)

        # 创建Excel工作簿
        workbook = xlsxwriter.Workbook(output_file, {'strings_to_numbers': False})
        worksheet = workbook.add_worksheet('Data')

        # 定义格式
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })

        text_format = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter'
        })

        # 写入标题行
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_format)
            # 设置列宽
            if col_name in ['GlobalEpoch', 'LocalEpoch', 'ChannelID', 'SatelliteID']:
                worksheet.set_column(col_idx, col_idx, 12)
            elif col_name == 'SignalType':
                worksheet.set_column(col_idx, col_idx, 12)
            else:
                # 科学计数法需要更宽的列
                worksheet.set_column(col_idx, col_idx, 18)

        # 确定哪些列需要转换为科学计数法
        numeric_cols = []
        for col in df.columns:
            if col not in ['SignalType', 'ChannelID', 'SatelliteID',
                          'GlobalEpoch', 'LocalEpoch']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)

        # 逐行写入数据
        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                value = row[col_name]

                # 数值列转换为科学计数法文本
                if col_name in numeric_cols:
                    if pd.notna(value):
                        # 转换为科学计数法字符串
                        text_value = f'{value:.10e}'
                        # 作为文本写入（加前缀空格强制为文本）
                        worksheet.write_string(row_idx + 1, col_idx, text_value, text_format)
                    else:
                        worksheet.write_string(row_idx + 1, col_idx, '', text_format)
                else:
                    # 非数值列直接写入
                    worksheet.write(row_idx + 1, col_idx, str(value), text_format)

            # 每10000行打印进度
            if (row_idx + 1) % 10000 == 0:
                print(f"  进度: {row_idx + 1}/{len(df)} 行")

        # 冻结首行
        worksheet.freeze_panes(1, 0)

        # 关闭工作簿
        workbook.close()

        print(f"  [OK] 已保存: {os.path.basename(output_file)}")
        print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
        return True

    except Exception as e:
        print(f"  [ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python csv_to_excel_text.py <csv_file_or_folder>")
        print("\n示例:")
        print("  python csv_to_excel_text.py data.csv")
        print("  python csv_to_excel_text.py D:\\data\\csv_output")
        sys.exit(1)

    path = sys.argv[1]

    print("\n" + "=" * 70)
    print("CSV to Excel Converter (Text Mode)")
    print("=" * 70)

    if os.path.isfile(path):
        # 单个文件
        if not path.lower().endswith('.csv'):
            print(f"错误: {path} 不是CSV文件")
            sys.exit(1)

        csv_to_excel_text(path)

    elif os.path.isdir(path):
        # 文件夹中的所有CSV
        csv_files = glob.glob(os.path.join(path, '*.csv'))

        if not csv_files:
            print(f"错误: 在 {path} 中未找到CSV文件")
            sys.exit(1)

        print(f"\n找到 {len(csv_files)} 个CSV文件\n")

        success_count = 0
        for csv_file in csv_files:
            if csv_to_excel_text(csv_file):
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
