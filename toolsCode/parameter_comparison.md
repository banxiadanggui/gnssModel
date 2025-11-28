# MAT文件参数导出对比分析

## 一、Python程序 (convert_utd_mat.py) 导出内容

### 导出文件：UTD_statResults.csv (105行 × 62列)

**类别1：位置信息 (Position)**
- X, Y, Z - ECEF坐标 (米)
- latitude, longitude, height - 大地坐标
- trueLat, trueLong, trueHeight - 真实坐标
- truex, truey, truez - 真实ECEF坐标
- meanx, meany, meanz - 平均ECEF坐标

**类别2：速度信息 (Velocity)**
- VX, VY, VZ - ECEF速度分量 (米/秒)
- vel_n, vel_e, vel_u - 北东天速度分量

**类别3：误差分析 (Error Statistics)**
- dx, dy, dz - 三维坐标误差
- dhor, dver - 水平/垂直误差
- RMSx, RMSy, RMSz, RMS3D - 均方根误差
- sn, se, su - 北东天误差
- truesn, truese, truesu - 真实北东天误差
- meansn, meanse, meansu - 平均北东天误差
- hor, ver - 水平/垂直精度指标

**类别4：时钟信息 (Clock)**
- dt - 接收机钟差 (秒)
- df - 接收机频偏 (Hz)

**类别5：其他 (Others)**
- res (17列) - 残差矩阵
- topo (3列) - 地形相关
- xr, yr, zr - 相对坐标
- Index50, Index95 - 精度指标
- dop - DOP值(几何精度因子)
- fom - 优值因子
- ppp - PPP相关

---

## 二、MATLAB程序 (extract_features.m) 导出内容

### 导出文件：Data_export.csv (行数不定)

**来源：obsData → channel级别数据**

导出的11个字段：
1. **carrierFreq** - 载波频率 (Hz)
2. **corrP** - 相关峰值
3. **trueRange** - 真实距离 (米)
4. **rangeResid** - 距离残差 (米)
5. **doppler** - 多普勒频移 (Hz)
6. **dopplerResid** - 多普勒残差 (Hz)
7. **tow** - GPS周内时 (秒)
8. **transmitTime** - 卫星发射时间
9. **satId** - 卫星ID
10. **receiverTow** - 接收机周内时
11. **signalName** - 信号类型 (如 "GPS_L1", "Galileo_E1")

---

## 三、重要但未导出的量

### 📊 从 statResults (Python已导出，MATLAB未涉及)
✓ Python已导出，包含定位结果和精度评估

### 📡 从 trackData (两者都未完整导出)

**重要跟踪参数：**
- **CN0** (C/N0) - 载噪比，信号质量的关键指标 ⚠️
- **PRN** - 伪随机码编号
- **lockIndicator** - 锁定指示器
- **pllDiscr** - 锁相环鉴相器输出
- **dllDiscr** - 延迟锁定环鉴相器输出
- **codePhase** - 码相位
- **carrierPhase** - 载波相位 ⚠️
- **I_P, Q_P** - 同相/正交提示积分值
- **I_E, Q_E, I_L, Q_L** - 超前/滞后积分值

### 🛰️ 从 ephData (两者都未导出)

**星历参数（对精密定位重要）：**
- **weekNumber** - GPS周数
- **t_oe** - 星历参考时间
- **sqrtA** - 轨道长半轴平方根
- **e** - 偏心率
- **omega_0, omega** - 升交点赤经、近地点角距
- **i_0** - 轨道倾角
- **M_0** - 平近点角
- **deltan** - 平均角速度修正
- **omegaDot, iDot** - 各项变化率
- **C_rs, C_rc, C_us, C_uc, C_is, C_ic** - 摄动修正项
- **a_f0, a_f1, a_f2** - 卫星钟差多项式系数 ⚠️
- **IODC, IODE** - 星历龄期
- **T_GD** - 群延迟

### 🌐 从 satData (两者都未导出)

**每历元的卫星信息：**
- **elevation** - 卫星高度角 ⚠️
- **azimuth** - 卫星方位角 ⚠️
- **satPos** - 卫星位置
- **satVel** - 卫星速度
- **satClkCorr** - 卫星钟差改正
- **travelTime** - 信号传播时间

### 📍 从 navData (两者都未导出)

**导航解算信息：**
- **usedSats** - 参与定位的卫星列表 ⚠️
- **nrSats** - 卫星数量 ⚠️
- **GDOP, PDOP, HDOP, VDOP, TDOP** - 各类DOP值 ⚠️
- **pos** - 位置解
- **vel** - 速度解
- **clockBias** - 钟差
- **clockDrift** - 钟漂

### ⚙️ 从 settings (两者都未导出)

**处理配置参数：**
- **samplingFreq** - 采样频率
- **IF** - 中频
- **codeFreqBasis** - 码频率基准
- **msToProcess** - 处理时长
- **elevation_mask** - 高度角截止角

### 🔍 从 acqData (两者都未导出)

**捕获数据：**
- **peakMetric** - 捕获峰值
- **codePhase** - 码相位
- **doppler** - 多普勒

---

## 四、重要性评估（按功能分类）

### 🔴 关键缺失参数（高优先级）

1. **CN0 (载噪比)** - 信号质量评估，干扰检测的核心指标
2. **卫星高度角/方位角** - 多路径效应分析、信号遮挡判断
3. **GDOP/PDOP/HDOP** - 几何精度因子，定位质量评估
4. **使用的卫星数量** - 定位可用性分析
5. **载波相位** - 高精度定位的关键观测量

### 🟡 重要参数（中优先级）

6. **锁相环/延迟锁定环鉴相器输出** - 跟踪性能分析
7. **I/Q积分值** - 信号相关性、干扰检测
8. **星历参数** - 卫星轨道和钟差信息
9. **卫星钟差改正** - 精密定位必需
10. **锁定指示器** - 跟踪状态判断

### 🟢 辅助参数（低优先级）

11. **采样频率、中频** - 数据处理配置
12. **捕获参数** - 初始化阶段信息
13. **处理时长** - 实验设置

---

## 五、数据完整性总结

| 数据类别 | Python程序 | MATLAB程序 | 重要性 |
|---------|-----------|-----------|--------|
| **定位结果** (statResults) | ✓ 已导出 | ✗ 未导出 | 🔴 极高 |
| **原始观测** (obsData) | ✗ 未导出 | ✓ 部分导出 | 🔴 高 |
| **信号跟踪** (trackData) | ✗ 未导出 | ✗ 未导出 | 🔴 高 |
| **卫星几何** (satData) | ✗ 未导出 | ✗ 未导出 | 🟡 中 |
| **导航解算** (navData) | ✗ 未导出 | ✗ 未导出 | 🟡 中 |
| **星历数据** (ephData) | ✗ 未导出 | ✗ 未导出 | 🟢 低-中 |
| **配置参数** (settings) | ✗ 未导出 | ✗ 未导出 | 🟢 低 |
| **捕获数据** (acqData) | ✗ 未导出 | ✗ 未导出 | 🟢 低 |

---

## 六、建议

### 对于干扰检测和分析，建议补充导出：
1. **trackData中的CN0** - 最关键的信号质量指标
2. **satData中的高度角/方位角** - 判断信号来源和几何关系
3. **navData中的DOP值和可见卫星数** - 定位几何质量
4. **trackData中的载波相位和I/Q值** - 深度信号分析

### 数据层次：
- **Python程序关注：** 高层次定位结果和统计信息（适合整体性能评估）
- **MATLAB程序关注：** 中层次观测值（适合观测数据分析）
- **缺失：** 底层信号跟踪细节（适合信号处理和干扰检测）
