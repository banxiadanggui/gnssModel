#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将GPS和Galileo跟踪数据合并到单个Excel文件的不同sheet中
所有数值列转换为科学计数法文本以避免WPS显示问题
"""

import pandas as pd
import xlsxwriter
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def write_dataframe_to_sheet(workbook, sheet_name, df, progress_label=""):
    """
    将DataFrame写入Excel sheet，所有数值列转为科学计数法文本

    Args:
        workbook: xlsxwriter工作簿对象
        sheet_name: sheet名称
        df: 要写入的DataFrame
        progress_label: 进度显示标签
    """
    print(f"\n处理 {progress_label}...")

    # 创建worksheet
    worksheet = workbook.add_worksheet(sheet_name)

    # 定义格式
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D3D3D3',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
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
            worksheet.set_column(col_idx, col_idx, 10)
        else:
            # 科学计数法需要更宽的列
            worksheet.set_column(col_idx, col_idx, 18)

    # 确定需要转换为科学计数法的列（所有数值列，除了元数据列）
    metadata_cols = {'SignalType', 'ChannelID', 'SatelliteID', 'GlobalEpoch', 'LocalEpoch'}
    scientific_cols = []

    for col in df.columns:
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col]):
            scientific_cols.append(col)

    print(f"  列数: {len(df.columns)}")
    print(f"  行数: {len(df)}")
    print(f"  科学计数法列: {len(scientific_cols)}")

    # 逐行写入数据
    for row_idx, row in df.iterrows():
        for col_idx, col_name in enumerate(df.columns):
            value = row[col_name]

            # 数值列转换为科学计数法文本
            if col_name in scientific_cols:
                if pd.notna(value):
                    # 转换为科学计数法字符串 (10位精度)
                    text_value = f'{float(value):.10e}'
                    worksheet.write_string(row_idx + 1, col_idx, text_value, text_format)
                else:
                    # NaN写为空字符串
                    worksheet.write_string(row_idx + 1, col_idx, '', text_format)
            else:
                # 元数据列直接写入
                if pd.notna(value):
                    worksheet.write(row_idx + 1, col_idx, str(value), text_format)
                else:
                    worksheet.write_string(row_idx + 1, col_idx, '', text_format)

        # 每10000行打印进度
        if (row_idx + 1) % 10000 == 0:
            print(f"  进度: {row_idx + 1}/{len(df)} 行")

    # 冻结首行
    worksheet.freeze_panes(1, 0)

    print(f"  [OK] 完成 {progress_label}")


def merge_trackdata_to_excel(csv_dir, output_file):
    """
    将GPS和Galileo CSV合并到单个Excel文件

    Args:
        csv_dir: CSV文件目录
        output_file: 输出Excel文件路径
    """
    print("=" * 80)
    print("合并GNSS跟踪数据到Excel (科学计数法文本格式)")
    print("=" * 80)

    # 定义输入文件
    gps_csv = os.path.join(csv_dir, "UTD_processed_latest_trackData_gpsl1.csv")
    gal_csv = os.path.join(csv_dir, "UTD_processed_latest_trackData_gale1b.csv")

    # 检查文件是否存在
    if not os.path.exists(gps_csv):
        print(f"[ERROR] GPS CSV文件不存在: {gps_csv}")
        return False

    if not os.path.exists(gal_csv):
        print(f"[ERROR] Galileo CSV文件不存在: {gal_csv}")
        return False

    print(f"\n输入文件:")
    print(f"  GPS L1:    {os.path.basename(gps_csv)}")
    print(f"  Galileo:   {os.path.basename(gal_csv)}")
    print(f"\n输出文件: {os.path.basename(output_file)}")

    try:
        # 读取CSV文件
        print("\n" + "-" * 80)
        print("加载CSV文件...")
        print("-" * 80)

        print("  加载GPS L1数据...")
        df_gps = pd.read_csv(gps_csv)
        print(f"    [OK] {len(df_gps)} 行 x {len(df_gps.columns)} 列")

        print("  加载Galileo E1B数据...")
        df_gal = pd.read_csv(gal_csv)
        print(f"    [OK] {len(df_gal)} 行 x {len(df_gal.columns)} 列")

        # 创建Excel工作簿
        print("\n" + "-" * 80)
        print("创建Excel工作簿...")
        print("-" * 80)

        workbook = xlsxwriter.Workbook(output_file, {
            'strings_to_numbers': False,  # 防止字符串被自动转换为数字
            'nan_inf_to_errors': True      # 将NaN/Inf转换为错误
        })

        # 写入GPS L1数据到第一个sheet
        write_dataframe_to_sheet(workbook, 'GPS_L1', df_gps, "GPS L1")

        # 写入Galileo E1B数据到第二个sheet
        write_dataframe_to_sheet(workbook, 'Galileo_E1B', df_gal, "Galileo E1B")

        # 关闭工作簿
        print("\n保存工作簿...")
        workbook.close()

        print("\n" + "=" * 80)
        print("[OK] 转换完成!")
        print("=" * 80)
        print(f"\n输出文件: {output_file}")
        print(f"  Sheet 1: GPS_L1      ({len(df_gps):,} 行)")
        print(f"  Sheet 2: Galileo_E1B ({len(df_gal):,} 行)")
        print()

        return True

    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 默认路径
    csv_dir = r"D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest"
    output_file = r"D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\GNSS_TrackData_Combined.xlsx"

    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) >= 2:
        csv_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    # 执行合并
    success = merge_trackdata_to_excel(csv_dir, output_file)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
