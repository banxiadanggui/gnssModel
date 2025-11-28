# CSV空行问题修复说明

## 问题描述

在原始版本中，导出的CSV文件存在**规律的空行**问题，特别是在跟踪数据和观测数据文件中。

## 根本原因

### 问题1：Epoch重复（主要原因）

**原始代码问题：**
```python
# 每个卫星通道的Epoch都从1开始
channel_dict = {
    'Epoch': np.arange(1, data_length + 1),  # ← 问题在这里！
    'ChannelID': np.full(data_length, ch_idx),
    'SatelliteID': np.full(data_length, sat_id),
}
```

**导致的问题：**
- CSV文件中有多个"Epoch 1", "Epoch 2"等重复值
- 13个GPS卫星 + 5个Galileo卫星 = 每个Epoch值重复18次
- 某些CSV查看器（如Excel）会将重复的Epoch显示为分组或空行
- 数据分析时难以区分不同卫星的同一时刻

**示例（原始CSV）：**
```
Epoch,ChannelID,SatelliteID,I_Prompt,...
1,1,2,0.123,...          # 卫星2的第1个历元
1,2,7,0.456,...          # 卫星7的第1个历元  ← Epoch重复！
1,3,8,0.789,...          # 卫星8的第1个历元  ← Epoch重复！
2,1,2,0.234,...
2,2,7,0.567,...          # ← Epoch继续重复
...
```

### 问题2：字段长度不匹配导致NaN值

**代码片段：**
```python
for mat_field, csv_field in field_mapping.items():
    field_data = get_struct_field(ch, mat_field)
    if field_data is not None:
        field_data = np.array(field_data).flatten()
        if len(field_data) == data_length:  # ← 严格检查长度
            channel_dict[csv_field] = field_data
        # 如果长度不匹配，字段被跳过，导致该列为NaN
```

**导致的问题：**
- 如果某个字段的数据长度与I_P不一致，该字段会被跳过
- 合并多个通道时，缺失字段会被pandas填充为NaN
- 在CSV中显示为空值

## 修复方案

### 修复1：添加全局Epoch（已修复）

**新代码：**
```python
global_epoch_offset = 0  # 全局Epoch偏移量

for ch_idx, ch in enumerate(channels, start=1):
    # ... 获取数据 ...

    channel_dict = {
        'GlobalEpoch': np.arange(global_epoch_offset + 1,
                                 global_epoch_offset + data_length + 1),
        'LocalEpoch': np.arange(1, data_length + 1),  # 保留本地Epoch
        'ChannelID': np.full(data_length, ch_idx),
        'SatelliteID': np.full(data_length, sat_id),
        'SignalType': [signal] * data_length,
    }

    global_epoch_offset += data_length  # 累加偏移量
```

**优点：**
- `GlobalEpoch`：全局唯一的行号，从1递增到总行数
- `LocalEpoch`：保留每个卫星通道内的本地时间索引
- 不再有重复值
- 便于数据排序和分析

**修复后的CSV：**
```
GlobalEpoch,LocalEpoch,ChannelID,SatelliteID,I_Prompt,...
1,1,1,2,0.123,...          # 全局第1行，卫星2的本地第1个历元
2,2,1,2,0.234,...          # 全局第2行，卫星2的本地第2个历元
...
50000,50000,1,2,0.999,...  # 卫星2的最后一行
50001,1,2,7,0.456,...      # 全局第50001行，卫星7的第1个历元
50002,2,2,7,0.567,...      # ← 全局Epoch唯一！
...
```

### 修复2：保留严格的长度检查（建议保持）

当前的长度检查机制实际上是**合理的**：
- 防止长度不匹配的数据被错误插入
- NaN值表示该时刻数据不可用，这是正常现象
- 建议保持现状

**如果需要更宽松的处理（不推荐）：**
```python
if len(field_data) == data_length:
    channel_dict[csv_field] = field_data
elif len(field_data) > 0:
    # 裁剪或填充到data_length
    if len(field_data) > data_length:
        channel_dict[csv_field] = field_data[:data_length]
    else:
        padded = np.full(data_length, np.nan)
        padded[:len(field_data)] = field_data
        channel_dict[csv_field] = padded
```

## 影响的文件

修复应用于以下导出函数：
- ✅ `export_track_data()` - 跟踪数据
- ✅ `export_obs_data()` - 观测数据
- ⚪ `export_nav_data()` - 导航数据（无此问题，单一时间序列）
- ⚪ `export_statistics()` - 统计结果（无此问题，汇总数据）

## 使用诊断工具

运行诊断脚本检查CSV文件：

```bash
python diagnose_csv.py output_trackData_gpsl1.csv
```

**输出示例：**
```
正在分析: output_trackData_gpsl1.csv

============================================================
总行数: 900000
总列数: 20

完全空行数: 0
包含空值的行数: 150000

GlobalEpoch值范围: 1 - 900000
GlobalEpoch最大重复次数: 1  ← 修复后无重复！

卫星通道分布:
   卫星 2: 50000 行
   卫星 7: 50000 行
   卫星 8: 50000 行
   ...
```

## 版本历史

- **v1.0** (2025-11-14) - 初始版本，存在Epoch重复问题
- **v1.1** (2025-11-14) - 修复Epoch重复问题，添加GlobalEpoch和LocalEpoch

## 建议

1. **重新转换数据**：使用修复后的脚本重新转换MAT文件
2. **使用GlobalEpoch排序**：数据分析时使用GlobalEpoch作为主键
3. **按卫星筛选**：使用SatelliteID筛选特定卫星的数据，然后使用LocalEpoch
4. **检查NaN值**：NaN值通常表示该时刻信号丢失或锁定失败，是正常现象

## 示例分析代码

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取修复后的CSV
df = pd.read_csv('UTD_processed_latest_trackData_gpsl1.csv')

# 按GlobalEpoch排序（确保数据连续）
df = df.sort_values('GlobalEpoch')

# 筛选特定卫星
sat_2 = df[df['SatelliteID'] == 2].copy()

# 使用LocalEpoch绘图（卫星内部时间）
plt.plot(sat_2['LocalEpoch'], sat_2['CNo_dBHz'])
plt.xlabel('Local Epoch')
plt.ylabel('C/N0 (dB-Hz)')
plt.title('Satellite 2 - C/N0 over time')
plt.show()

# 或使用GlobalEpoch查看所有卫星
plt.figure(figsize=(15, 5))
for sat_id in df['SatelliteID'].unique():
    sat_data = df[df['SatelliteID'] == sat_id]
    plt.plot(sat_data['GlobalEpoch'], sat_data['CNo_dBHz'],
             label=f'Sat {sat_id}', alpha=0.7)
plt.xlabel('Global Epoch')
plt.ylabel('C/N0 (dB-Hz)')
plt.legend()
plt.show()
```
