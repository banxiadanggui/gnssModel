#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查MAT文件包含的变量"""

import scipy.io as sio
import sys
import io

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载MAT文件
mat_file = r'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat'
print(f"正在检查: {mat_file}\n")

try:
    data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

    # 获取所有非系统变量
    variables = [k for k in data.keys() if not k.startswith('__')]

    print("=" * 60)
    print("MAT文件包含的顶级变量:")
    print("=" * 60)

    for var in sorted(variables):
        var_type = type(data[var]).__name__
        print(f"  [OK] {var:<20} ({var_type})")

    print("\n" + "=" * 60)
    print("详细检查:")
    print("=" * 60)

    # 检查obsData
    if 'obsData' in variables:
        obs = data['obsData']
        print("\n[OK] obsData 存在")
        if hasattr(obs, '_fieldnames'):
            print(f"  包含的信号: {obs._fieldnames}")
        else:
            print(f"  结构: {dir(obs)}")
    else:
        print("\n[X] obsData 不存在")

    # 检查navData
    if 'navData' in variables:
        nav = data['navData']
        print("\n[OK] navData 存在")
        if hasattr(nav, '_fieldnames'):
            print(f"  包含的字段: {nav._fieldnames}")
        else:
            print(f"  结构: {dir(nav)}")
    else:
        print("\n[X] navData 不存在")

    # 检查trackData
    if 'trackData' in variables:
        track = data['trackData']
        print("\n[OK] trackData 存在")
        if hasattr(track, '_fieldnames'):
            print(f"  包含的信号: {track._fieldnames}")
    else:
        print("\n[X] trackData 不存在")

    print("\n" + "=" * 60)

except Exception as e:
    print(f"错误: {e}")
