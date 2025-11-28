#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查trackData包含的字段"""

import scipy.io as sio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载MAT文件
mat_file = r'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat'
print(f"加载: {mat_file}\n")

data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=False)
trackData = data['trackData']

# 展开单元素数组
if hasattr(trackData, 'flatten'):
    trackData = trackData.flatten()[0]

print("=" * 70)
print("trackData 结构:")
print("=" * 70)

# 查看信号
if hasattr(trackData, '_fieldnames'):
    signals = trackData._fieldnames
    print(f"包含的信号: {signals}\n")

    for signal in signals:
        if signal == 'trackingRunTime':
            continue

        print(f"\n信号: {signal}")
        print("-" * 70)

        signal_data = getattr(trackData, signal)
        if hasattr(signal_data, 'flatten'):
            signal_data = signal_data.flatten()[0]

        if hasattr(signal_data, 'channel'):
            channels = signal_data.channel
            if hasattr(channels, 'flatten'):
                channels = channels.flatten()

            print(f"  通道数: {len(channels)}")

            # 查看第一个通道的字段
            if len(channels) > 0:
                ch = channels[0]
                print(f"  第一个通道包含的字段:")

                if hasattr(ch, '_fieldnames'):
                    fields = ch._fieldnames
                    for field in fields:
                        field_data = getattr(ch, field)
                        if hasattr(field_data, 'shape'):
                            print(f"    {field:<25} shape={field_data.shape}, dtype={field_data.dtype}")
                        else:
                            print(f"    {field:<25} type={type(field_data).__name__}")
                else:
                    print(f"    (无法访问字段列表)")
                    print(f"    通道类型: {type(ch)}")
                    if hasattr(ch, '__dict__'):
                        print(f"    属性: {list(ch.__dict__.keys())}")
