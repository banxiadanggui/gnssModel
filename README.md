# GNSS信号处理与定位系统

基于FGI-GSRx的多星座GNSS（GPS/北斗/Galileo）软件接收机，支持信号捕获、跟踪、导航电文解码和定位解算。

## 项目概述

本项目是一个完整的GNSS信号处理系统，能够处理原始射频信号数据，进行信号捕获、跟踪、解码，并计算接收机位置。支持多个GNSS星座（GPS L1/L5、北斗B1、Galileo E1/E5等）的单独或组合定位。

**核心功能**：
- GNSS信号捕获（Acquisition）
- 多通道并行信号跟踪（Tracking）
- 导航电文解码（Frame Decoding）
- 定位解算（Navigation Solution）
- 数据导出与分析（MAT/CSV/Excel格式）

**技术特点**：
- 支持并行处理（多核CPU加速3-5倍）
- 支持多星座组合定位
- 完整的数据导出工具链
- 丰富的可视化功能

## 项目结构

```
beidou/
├── matlabFGI_ori/              # 原始FGI-GSRx代码库
│   ├── acq/                    # 信号捕获模块
│   ├── track/                  # 信号跟踪模块
│   ├── frame/                  # 导航电文解码模块
│   ├── nav/                    # 定位解算模块
│   ├── obs/                    # 观测量处理模块
│   ├── param/                  # 配置文件
│   └── main/                   # 主程序入口
│
├── matlabFGI_shared/           # 改进版FGI-GSRx（支持并行处理）
│   ├── acq/                    # 信号捕获（改进版）
│   ├── track/                  # 并行信号跟踪
│   ├── frame/                  # 导航电文解码
│   ├── nav/                    # 定位解算
│   ├── param/                  # 配置文件（包含并行处理配置）
│   └── main/                   # 主程序入口 gsrx.m
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始GNSS信号数据（.dat格式）
│   │   ├── finlandF1E1_dat/    # 芬兰L1+E1数据集
│   │   ├── finlandF5E5_dat/    # 芬兰L5+E5数据集
│   │   ├── jammer_iq/          # 干扰数据
│   │   └── sta_dat/            # 静态数据
│   ├── processedbat/           # 批处理配置和临时文件
│   │   └── parallel/           # 并行处理的跟踪结果（按卫星分文件）
│   ├── processedMAT/           # 处理后的完整结果（.mat格式）
│   │   └── temp_parallel/      # 并行处理临时文件
│   ├── processedCSV/           # 转换为CSV的处理结果
│   └── originalresults/        # 原始结果备份
│
├── processedData/              # 最终定位结果（CSV格式）
│   ├── *_FINAL_SOLUTION.csv    # 各数据集的定位结果
│   └── ...
│
├── toolsCode/                  # 数据转换和分析工具
│   ├── gnss_mat_to_csv.py     # MAT文件转CSV主工具
│   ├── mat_to_csv_converter.py # MAT转CSV转换器
│   ├── csv_to_excel.py        # CSV转Excel工具
│   ├── analyze_data_values.py  # 数据分析工具
│   ├── extract_features_enhanced.m  # 特征提取（MATLAB）
│   └── ...
│
├── recordPicture/              # 处理结果图片
├── relatedPaper/               # 相关论文和文档
├── readme/                     # 原始文档
│   └── README.pdf
└── README.md                   # 本文件

```

## 环境要求

### MATLAB环境
- MATLAB R2018b 或更高版本
- 推荐安装工具箱：
  - Signal Processing Toolbox
  - Communications Toolbox
  - Parallel Computing Toolbox（用于并行处理）

### Python环境
- Python 3.6+
- 必需的包：
  ```bash
  pip install numpy pandas scipy h5py openpyxl
  ```

### 硬件建议
- CPU: Intel i7或更高（8核以上推荐，用于并行处理）
- RAM: 16GB以上
- 存储: 至少50GB可用空间（用于存储原始数据和处理结果）

## 运行命令

### 1. GNSS信号处理（MATLAB）

#### 基本运行（单次处理）
```matlab
% 启动MATLAB，进入项目目录
cd D:\skill\beidou\matlabFGI_shared\main

% 运行接收机主程序
gsrx('test_UTD_share_parallel.txt')
```

#### 常用配置文件
```matlab
% GPS L1 处理
gsrx('test_param_FGISpoofRepo_GPSL1_UTD.txt')

% Galileo E1 处理
gsrx('test_param_FGISpoofRepo_GalE1_UTD.txt')

% GPS L5 + Galileo E5a 双频处理
gsrx('test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt')

% 并行处理配置（推荐，速度快3-5倍）
gsrx('test_UTD_share_parallel.txt')
```

#### 批处理运行
```bash
# Windows批处理（在data/processedbat/目录下）
ParallelProcess.bat
```

### 2. 数据转换与导出（Python）

#### MAT转CSV转换
```bash
# 基本用法：转换单个MAT文件
python toolsCode/gnss_mat_to_csv.py <mat_file_path>

# 示例：转换UTD处理结果
python toolsCode/gnss_mat_to_csv.py data/processedMAT/UTD_processed_latest.mat

# 指定输出目录
python toolsCode/gnss_mat_to_csv.py data/processedMAT/UTD_processed_latest.mat data/processedCSV/
```

#### CSV转Excel
```bash
# 将CSV数据整合到Excel表格
python toolsCode/csv_to_excel.py

# 使用文本格式（避免科学计数法）
python toolsCode/csv_to_excel_text.py
```

#### 数据分析
```bash
# 分析数据值分布
python toolsCode/analyze_data_values.py

# 检查数据结构
python toolsCode/check_mat_content.py

# 诊断CSV文件
python toolsCode/diagnose_csv.py
```

### 3. 特征提取（MATLAB）

```bash
# 使用批处理运行
toolsCode/run_extract_features.bat

# 或在MATLAB中运行
matlab -batch "cd('D:\skill\beidou\toolsCode'); extract_features_enhanced; exit"
```

## 数据结构

### 原始输入数据

#### GNSS信号数据文件（.dat格式）
- **文件位置**: `data/raw/`
- **数据类型**: 二进制原始I/Q采样数据
- **采样格式**: 通常为int8或int16，I和Q交替存储
- **采样率**: 典型值 4-40 MHz
- **数据集示例**:
  - `finlandF1E1_dat/`: GPS L1 + Galileo E1（中心频率1575.42 MHz）
  - `finlandF5E5_dat/`: GPS L5 + Galileo E5a（中心频率1176.45 MHz）

### 中间处理数据

#### 并行跟踪结果（.mat格式）
- **文件位置**: `data/processedbat/parallel/`
- **命名格式**: `trackData_<signal>_Satellite_ID_<id>.mat`
- **数据内容**: 单个卫星的跟踪数据
  - 相关器输出（I_Prompt, Q_Prompt, I_Early, Q_Early, I_Late, Q_Late）
  - 载波相位和频率
  - 码相位和码频率
  - 载噪比（C/N0）
  - 锁定指示器（PLL_Lock, DLL_Lock, FLL_Lock）

#### 完整处理结果（.mat格式）
- **文件位置**: `data/processedMAT/`
- **文件名**: `UTD_processed_latest.mat` 或带时间戳的文件名
- **数据结构**:
  ```matlab
  % 主要变量结构
  acqData         % 捕获结果
    ├─ signal     % 信号类型（gpsl1, gale1b等）
    ├─ channel()  % 通道数组
    │   ├─ sv    % 卫星编号
    │   ├─ doppler  % 多普勒频偏
    │   └─ codePhase  % 码相位

  trackData       % 跟踪结果
    ├─ signal     % 信号类型
    ├─ channel()  % 通道数组
    │   ├─ sv    % 卫星编号
    │   ├─ IP    % Prompt I相关器输出
    │   ├─ QP    % Prompt Q相关器输出
    │   ├─ IE, QE, IL, QL  % Early/Late相关器
    │   ├─ carrPhase  % 载波相位
    │   ├─ codePhase  % 码相位
    │   └─ CNo    % 载噪比

  navData         % 导航结果
    ├─ position   % 位置信息
    │   ├─ X, Y, Z        % ECEF坐标（米）
    │   ├─ latitude       % 纬度（度）
    │   ├─ longitude      % 经度（度）
    │   ├─ altitude       % 高度（米）
    │   └─ clockBias      % 钟差（秒）
    ├─ velocity   % 速度信息
    │   ├─ velX, velY, velZ  % ECEF速度（米/秒）
    └─ DOP       % 精度因子
        ├─ GDOP, PDOP, HDOP, VDOP, TDOP

  obsData         % 观测数据
    ├─ signal
    ├─ channel()
    │   ├─ pseudorange     % 伪距（米）
    │   ├─ carrierPhase    % 载波相位（周）
    │   ├─ doppler         % 多普勒（Hz）
    │   └─ CNo            % 载噪比（dB-Hz）

  satData         % 卫星数据
    ├─ signal
    ├─ channel()
    │   ├─ ephemeris       % 星历参数
    │   └─ clockCorrection % 卫星钟差修正

  statResults     % 统计结果
    ├─ horizontal_mean_m   % 水平误差均值
    ├─ horizontal_std_m    % 水平误差标准差
    ├─ horizontal_rms_m    % 水平RMS误差
    ├─ vertical_rms_m      % 垂直RMS误差
    └─ rms_3d_m           % 3D RMS误差
  ```

### 导出数据（CSV格式）

#### 跟踪数据文件
- **文件位置**: `data/processedCSV/`
- **命名格式**: `<filename>_trackData_<signal>.csv`
- **主要字段**:
  | 字段名 | 单位 | 说明 |
  |--------|------|------|
  | Epoch | - | 历元编号 |
  | ChannelID | - | 通道ID |
  | SatelliteID | - | 卫星PRN号 |
  | I_Prompt, Q_Prompt | - | 提示相关器I/Q值 |
  | I_Early, Q_Early | - | 超前相关器I/Q值 |
  | I_Late, Q_Late | - | 滞后相关器I/Q值 |
  | CarrierPhase_rad | rad | 载波相位 |
  | CarrierFreq_Hz | Hz | 载波频率 |
  | CodePhase_chips | chips | 码相位 |
  | CodeFreq_Hz | Hz | 码频率 |
  | CNo_dBHz | dB-Hz | 载噪比 |

#### 导航位置数据
- **命名格式**: `<filename>_navData_Position.csv`
- **主要字段**:
  | 字段名 | 单位 | 说明 |
  |--------|------|------|
  | Epoch | - | 历元编号 |
  | X_m, Y_m, Z_m | m | ECEF坐标 |
  | Latitude_deg | deg | 纬度 |
  | Longitude_deg | deg | 经度 |
  | Altitude_m | m | 高度 |
  | ClockBias_s | s | 接收机钟差 |
  | NumSatellites | - | 可用卫星数 |
  | GDOP, PDOP, HDOP, VDOP | - | 精度因子 |

#### 观测数据文件
- **命名格式**: `<filename>_obsData_<signal>.csv`
- **主要字段**:
  | 字段名 | 单位 | 说明 |
  |--------|------|------|
  | Epoch | - | 历元编号 |
  | ChannelID | - | 通道ID |
  | SatelliteID | - | 卫星PRN号 |
  | Pseudorange_m | m | 原始伪距 |
  | CorrectedPseudorange_m | m | 修正后伪距 |
  | CarrierPhase_cycles | cycles | 载波相位 |
  | Doppler_Hz | Hz | 多普勒频移 |
  | CNo_dBHz | dB-Hz | 载噪比 |

### 最终定位结果

#### 定位解算结果（CSV格式）
- **文件位置**: `processedData/`
- **命名格式**: `<dataset>_<signal>_FINAL_SOLUTION.csv`
- **数据内容**: 时序定位结果，包含位置、速度、精度信息

## 代码功能

### MATLAB核心模块（matlabFGI_shared/）

#### 1. 主程序模块（main/）
- **gsrx.m**: FGI-GSRx接收机主程序
  - 功能：协调整个信号处理流程
  - 输入：配置文件（.txt）
  - 输出：完整处理结果（.mat）
  - 流程：初始化 → 捕获 → 跟踪 → 解码 → 定位 → 保存

#### 2. 信号捕获模块（acq/）
- **doAcquisition.m**: 执行信号捕获
  - 功能：搜索可见卫星，粗略估计多普勒频偏和码相位
  - 算法：并行码相位搜索（PCPS）
  - 输出：捕获到的卫星列表及其初始参数

#### 3. 信号跟踪模块（track/）
- **doTrackingSingleChannel.m**: 单通道信号跟踪
  - 功能：精确跟踪单颗卫星信号
  - 环路：载波跟踪（PLL/FLL）+ 码跟踪（DLL）
  - 支持：多相关器跟踪（提高精度）

- **并行处理支持**:
  - 每颗卫星独立MATLAB进程
  - 临时文件存储在 `temp_parallel/`
  - 最终合并所有跟踪结果

#### 4. 导航电文解码模块（frame/）
- **doFrameDecoding.m**: 导航电文解码主函数
- **beib1DecodeEphemeris.m**: 北斗B1星历解码
- **gpsl1DecodeEphemeris.m**: GPS L1星历解码
- **gale1bDecodeEphemeris.m**: Galileo E1星历解码
- 功能：提取卫星星历、钟差参数、电离层参数等

#### 5. 定位解算模块（nav/）
- **doNavigation.m**: 导航定位主函数
- **calcPosLSE.m**: 最小二乘定位算法
- **calcVelLSE.m**: 速度估计
- **calculatePseudoRanges.m**: 伪距计算
- 功能：基于伪距观测值计算接收机位置和速度

#### 6. 观测量处理模块（obs/）
- **generateObservations.m**: 生成观测量
- **applyIonoCorrections.m**: 电离层延迟修正
- **applyTropoCorrections.m**: 对流层延迟修正
- **applyClockCorrections.m**: 卫星钟差修正
- 功能：生成和修正各类GNSS观测量

#### 7. 辅助模块
- **param/**: 参数配置文件和读取函数
- **plot/**: 绘图函数（捕获图、跟踪图、定位轨迹等）
- **geo/**: 坐标转换函数（ECEF ↔ 大地坐标）
- **time/**: GNSS时间系统转换
- **sat/**: 卫星位置计算
- **stats/**: 统计分析工具

### Python工具模块（toolsCode/）

#### 1. 数据转换工具

**gnss_mat_to_csv.py** - MAT文件转CSV主工具
- 功能：将MATLAB处理结果转换为CSV格式
- 支持：
  - MATLAB v7.2格式（scipy）
  - MATLAB v7.3格式（h5py）
- 输出：
  - 跟踪数据CSV（按信号类型）
  - 导航位置CSV
  - 导航速度CSV
  - 观测数据CSV（按信号类型）
  - 统计结果CSV

**mat_to_csv_converter.py** - 通用MAT转换器
- 功能：通用MAT文件读取和转换
- 特性：自动处理嵌套结构体、数组展平

**csv_to_excel.py / csv_to_excel_text.py** - CSV转Excel工具
- 功能：将多个CSV文件整合到一个Excel文件
- 差异：text版本使用文本格式避免科学计数法

#### 2. 数据分析工具

**analyze_data_values.py** - 数据值分布分析
- 功能：统计各字段的数值范围、缺失值情况

**check_mat_content.py** - MAT文件内容检查
- 功能：快速查看MAT文件包含的变量和结构

**diagnose_csv.py** - CSV文件诊断
- 功能：检查CSV文件完整性和数据质量

**compare_mat_excel_fields.py** - 字段对比工具
- 功能：对比MAT和Excel文件的字段差异

#### 3. MATLAB辅助工具

**extract_features_enhanced.m** - 特征提取工具
- 功能：从跟踪结果中提取关键特征
- 输出：特征向量CSV文件

**check_satdata_navdata.m** - 数据检查工具
- 功能：检查卫星数据和导航数据的完整性

**inspect_mat_fields.m** - MAT结构检查
- 功能：显示MAT文件的详细结构信息

## 典型工作流程

### 完整处理流程

```
1. 准备原始数据
   └─ 放置.dat信号数据到 data/raw/

2. 配置参数文件
   └─ 修改 matlabFGI_shared/param/test_UTD_share_parallel.txt
      ├─ 设置数据文件路径
      ├─ 选择启用的信号类型
      ├─ 配置并行处理参数
      └─ 设置输出文件路径

3. 运行MATLAB信号处理
   └─ 命令: gsrx('test_UTD_share_parallel.txt')
      ├─ 信号捕获（约5-10分钟）
      ├─ 并行跟踪（约20-40分钟）
      ├─ 电文解码（约5分钟）
      ├─ 定位解算（约2分钟）
      └─ 保存结果 → data/processedMAT/UTD_processed_latest.mat

4. 转换为CSV格式
   └─ 命令: python toolsCode/gnss_mat_to_csv.py data/processedMAT/UTD_processed_latest.mat
      └─ 输出 → data/processedCSV/
          ├─ UTD_processed_latest_trackData_gpsl1.csv
          ├─ UTD_processed_latest_trackData_gale1b.csv
          ├─ UTD_processed_latest_navData_Position.csv
          ├─ UTD_processed_latest_navData_Velocity.csv
          ├─ UTD_processed_latest_obsData_gpsl1.csv
          ├─ UTD_processed_latest_obsData_gale1b.csv
          └─ UTD_processed_latest_statistics.csv

5. （可选）生成Excel报告
   └─ 命令: python toolsCode/csv_to_excel_text.py
      └─ 输出 → processedData/<dataset>_analysis.xlsx

6. 数据分析与可视化
   └─ 使用Python/MATLAB进行进一步分析
```

### 快速测试流程（已有处理结果）

如果已有MAT处理结果，可以直接进行数据转换和分析：

```bash
# 1. 转换为CSV
python toolsCode/gnss_mat_to_csv.py data/processedMAT/UTD_processed_latest.mat

# 2. 生成Excel报告
python toolsCode/csv_to_excel_text.py

# 3. 数据分析
python toolsCode/analyze_data_values.py
```

## 常见问题

### 1. MATLAB运行错误

**问题**: 找不到函数或路径错误
**解决**: 确保正确设置MATLAB路径
```matlab
addpath(genpath('D:\skill\beidou\matlabFGI_shared'))
```

**问题**: 并行处理失败
**解决**: 检查临时文件目录是否存在
```matlab
mkdir('D:\skill\beidou\data\processedMAT\temp_parallel')
```

### 2. Python转换错误

**问题**: 无法读取MAT文件
**解决**: 升级依赖库
```bash
pip install --upgrade scipy h5py numpy pandas
```

**问题**: 输出CSV包含大量NaN
**解决**: 这是正常现象，表示某些历元数据不可用

### 3. 数据处理建议

- **首次运行**: 建议使用较短的数据（设置 `sys,msToProcess` 为较小值，如60000=1分钟）
- **并行处理**: 确保CPU核心数充足，避免设置过多并行通道
- **内存不足**: 可以关闭绘图功能（`sys,plotSpectra=false`）来节省内存

## 版本信息

- **Git仓库**: 已初始化
- **当前分支**: main
- **最近提交**:
  - f85764c: 上一条的补充
  - 4190ceb: 本地跑通的平行改进版本
  - 0c80d5d: 整理文件夹与工具代码

## 相关资源

- **FGI-GSRx官方文档**: readme/README.pdf
- **配置文件示例**: matlabFGI_shared/param/
- **工具文档**: toolsCode/README_gnss_mat_to_csv.md
- **相关论文**: relatedPaper/

## 许可证

本项目基于FGI-GSRx开源软件接收机开发，遵循GNU General Public License v3.0。
FGI-GSRx版权归Finnish Geospatial Research Institute (FGI)所有。

## 联系方式

项目路径: D:\skill\beidou\
工作目录: D:\skill\beidou\matlabFGI_shared\

---

**更新日期**: 2025-11-28
**文档版本**: v1.0
