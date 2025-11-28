# GNSS MAT to CSV Converter

将GNSS接收机处理后的MAT文件转换为CSV格式的工具。

## 功能特性

- 支持MATLAB v7.2及以下格式（使用scipy）
- 支持MATLAB v7.3 HDF5格式（使用h5py）
- 自动导出多种数据类型：
  - **跟踪数据** (trackData) - 每个信号类型单独一个CSV文件
  - **导航数据** (navData) - 位置和速度数据
  - **观测数据** (obsData) - 伪距、载波相位、多普勒等
  - **统计结果** (statResults) - 定位精度统计

## 环境要求

### Python版本
- Python 3.6+

### 依赖库

```bash
pip install numpy pandas scipy h5py
```

或者使用requirements.txt（如果提供）：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python gnss_mat_to_csv.py <mat_file_path>
```

### 指定输出目录

```bash
python gnss_mat_to_csv.py <mat_file_path> <output_dir>
```

### 示例

#### 示例 1：转换UTD数据集处理结果

```bash
python gnss_mat_to_csv.py D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat
```

输出目录：`D:\skill\beidou\data\processedMAT\csv_output\UTD_processed_latest\`

#### 示例 2：指定输出目录

```bash
python gnss_mat_to_csv.py D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat D:\output\csv_files
```

输出目录：`D:\output\csv_files\`

## 输出文件说明

### 1. 跟踪数据文件 (`*_trackData_SIGNAL.csv`)

每个GNSS信号类型（如gpsl1, gale1b）生成一个文件。

**包含字段：**
- `Epoch` - 历元编号
- `ChannelID` - 通道ID
- `SatelliteID` - 卫星编号
- `SignalType` - 信号类型
- `I_Prompt`, `Q_Prompt` - 提示相关器I/Q值
- `I_Early`, `Q_Early` - 超前相关器I/Q值
- `I_Late`, `Q_Late` - 滞后相关器I/Q值
- `CarrierPhase_rad` - 载波相位（弧度）
- `CarrierFreq_Hz` - 载波频率（Hz）
- `CarrierFreqRate_Hz_s` - 载波频率变化率（Hz/s）
- `CodePhase_chips` - 码相位（码片）
- `CodeFreq_Hz` - 码频率（Hz）
- `CNo_dBHz` - 载噪比（dB-Hz）
- `PLL_Lock`, `FLL_Lock`, `DLL_Lock` - 锁定指示器

**示例：**
```
UTD_processed_latest_trackData_gpsl1.csv
UTD_processed_latest_trackData_gale1b.csv
```

### 2. 导航位置数据 (`*_navData_Position.csv`)

**包含字段：**
- `Epoch` - 历元编号
- `X_m`, `Y_m`, `Z_m` - ECEF坐标（米）
- `Latitude_deg`, `Longitude_deg`, `Altitude_m` - 大地坐标
- `ClockBias_s` - 接收机钟差（秒）
- `NumSatellites` - 可用卫星数
- `FOM` - 精度因子
- `GDOP`, `PDOP`, `HDOP`, `VDOP`, `TDOP` - DOP值

**示例：**
```
UTD_processed_latest_navData_Position.csv
```

### 3. 导航速度数据 (`*_navData_Velocity.csv`)

**包含字段：**
- `Epoch` - 历元编号
- `VelX_m_s`, `VelY_m_s`, `VelZ_m_s` - ECEF速度（米/秒）
- `VelFOM` - 速度精度因子

**示例：**
```
UTD_processed_latest_navData_Velocity.csv
```

### 4. 观测数据文件 (`*_obsData_SIGNAL.csv`)

每个GNSS信号类型生成一个文件。

**包含字段：**
- `Epoch` - 历元编号
- `ChannelID` - 通道ID
- `SatelliteID` - 卫星编号
- `Pseudorange_m` - 原始伪距（米）
- `CorrectedPseudorange_m` - 修正后伪距（米）
- `CarrierPhase_cycles` - 载波相位（周）
- `Doppler_Hz` - 多普勒频移（Hz）
- `CNo_dBHz` - 载噪比（dB-Hz）

**示例：**
```
UTD_processed_latest_obsData_gpsl1.csv
UTD_processed_latest_obsData_gale1b.csv
```

### 5. 统计结果 (`*_statistics.csv`)

定位精度统计信息。

**包含字段：**
- `Horizontal_Mean_m`, `Horizontal_Std_m`, `Horizontal_RMS_m`, `Horizontal_95th_m` - 水平误差统计
- `Vertical_Mean_m`, `Vertical_Std_m`, `Vertical_RMS_m`, `Vertical_95th_m` - 垂直误差统计
- `RMS_3D_m` - 3D RMS误差
- `Mean_GDOP`, `Mean_PDOP`, `Mean_HDOP`, `Mean_VDOP`, `Mean_TDOP` - 平均DOP值

**示例：**
```
UTD_processed_latest_statistics.csv
```

## 数据分析建议

### 使用Python Pandas读取

```python
import pandas as pd

# 读取跟踪数据
track_data = pd.read_csv('UTD_processed_latest_trackData_gpsl1.csv')

# 读取导航位置
nav_pos = pd.read_csv('UTD_processed_latest_navData_Position.csv')

# 绘制定位轨迹
import matplotlib.pyplot as plt
plt.scatter(nav_pos['Longitude_deg'], nav_pos['Latitude_deg'])
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.show()
```

### 使用Excel分析

直接双击CSV文件即可在Excel中打开进行分析。

## 故障排除

### 问题1：提示"无法加载MAT文件"

**解决方法：**
```bash
pip install --upgrade scipy h5py
```

### 问题2：某些信号数据未导出

**原因：** MAT文件中可能不包含该信号的数据。

**检查方法：**
- 查看转换输出日志中的警告信息
- 检查MAT文件是否包含对应信号的跟踪结果

### 问题3：CSV文件中有大量NaN值

**原因：** 某些通道或历元的数据缺失。

**说明：** 这是正常现象，表示该时刻数据不可用。

## 技术支持

如有问题或建议，请联系：
- 项目路径：`D:\skill\beidou\toolsCode\`
- 脚本文件：`gnss_mat_to_csv.py`

## 版本历史

- **v1.0** (2025-11-14)
  - 初始版本
  - 支持trackData, navData, obsData, statResults导出
  - 支持MATLAB v7.2和v7.3格式

## 许可证

本工具基于GNSS数据处理工具箱开发，仅供研究和教育使用。
