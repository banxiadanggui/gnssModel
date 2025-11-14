# GPS L5 / Galileo E5a Signal Processing Module

## 概述

这是FGI-GSRx的独立扩展模块，用于处理GPS L5和Galileo E5a信号。该模块不修改原有代码，作为独立系统运行。

## 创建的文件

### 主程序
- **`main/gsrx_L5.m`**: L5/E5a专用主程序

### 参数处理
- **`param/getSystemParameters_L5.m`**: L5/E5a参数处理函数
- **`param/readSettings_L5.m`**: L5/E5a配置读取函数
- **`param/test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt`**: L5/E5a参数配置文件

### 信号处理
- **`mod/gpsl5GeneratePrnCode.m`**: GPS L5 PRN码生成器 (10230 chips)
- **`mod/gale5aGeneratePrnCode.m`**: Galileo E5a PRN码生成器 (10230 chips)

## 技术参数

### GPS L5
| 参数 | 值 |
|------|-----|
| 载波频率 | 1176.45 MHz |
| 码速率 | 10.23 Mcps |
| 码长度 | 10230 chips (1 ms) |
| 调制方式 | BPSK(10) |
| 带宽 | ~20 MHz |
| PRN范围 | 1-32 |

### Galileo E5a
| 参数 | 值 |
|------|-----|
| 载波频率 | 1176.45 MHz |
| 码速率 | 10.23 Mcps |
| 码长度 | 10230 chips (1 ms) |
| 调制方式 | AltBOC(15,10) |
| 带宽 | ~20 MHz |
| PRN范围 | 1-36 |

## 使用方法

### 1. 准备RF数据文件
确保你有L5/E5a频段的RF数据文件：
- 中心频率：1176.45 MHz
- 采样率：≥ 26 MSPS
- 数据格式：复数采样 (I/Q)

### 2. 配置参数文件
编辑 `param/test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt`：

```matlab
% 设置信号类型
sys,enabledSignals,[{['gpsl5']} {['gale5a']}],

% 设置数据文件路径
gpsl5,rfFileName,'你的数据文件路径.dat',
gale5a,rfFileName,'你的数据文件路径.dat',

% 设置射频前端参数
gpsl5,centerFrequency,1176.45e6,
gpsl5,samplingFreq,26e6,
```

### 3. 运行主程序
在MATLAB中：

```matlab
cd('D:\skill\beidou\FGI-GSRx-2.0.2\main')
gsrx_L5('..\param\test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt')
```

## 当前状态与限制

### ⚠️ 重要提示

这是一个**框架实现**，提供了基本结构但**尚未完全功能化**。需要额外工作：

### 需要实现的功能

#### 1. 捕获模块
需要创建 `doAcquisition_L5.m`：
- L5/E5a信号的FFT捕获
- 考虑更宽的频率搜索范围
- 非相干积分 (10次)
- 相干积分 (10ms)

#### 2. 跟踪模块
需要创建 `doTracking_L5.m`：
- FLL/PLL载波跟踪环路
- DLL码跟踪环路
- L5特定的环路带宽参数
- 10230 chips码周期处理

#### 3. 帧解码模块
需要创建 `doFrameDecoding_L5.m`：
- GPS L5 CNAV message解码
- Galileo E5a I/NAV message解码
- 二次码处理
- 时间戳提取

#### 4. PRN码调制
需要创建：
- `gpsl5ModulatePrnCode.m`
- `gale5aModulatePrnCode.m`

参考现有的：
- `gpsl1ModulatePrnCode.m`
- `gale1bModulatePrnCode.m`

## PRN码生成器说明

### GPS L5 (`gpsl5GeneratePrnCode.m`)

使用13位LFSR生成10230-chip代码：
- XA寄存器：反馈多项式 1 + X^9 + X^10 + X^12 + X^13
- XB寄存器：PRN特定初始状态
- 输出：XA(13) XOR XB(13)

### Galileo E5a (`gale5aGeneratePrnCode.m`)

结构类似GPS L5：
- 使用13位LFSR
- PRN特定初始状态（需要根据Galileo ICD验证）
- 10230 chips输出

**注意**: E5a的初始状态值是示例，需要查阅Galileo OS SIS ICD获取准确值。

## 测试建议

### 1. PRN码验证
```matlab
% 测试GPS L5 PRN码生成
code = gpsl5GeneratePrnCode(1);
length(code)  % 应该是 10230
sum(code == 1) + sum(code == -1)  % 应该等于 10230

% 测试自相关
autocorr = xcorr(code);
max(autocorr)  % 峰值应该在中心
```

### 2. 参数加载测试
```matlab
% 测试参数文件读取
settings = readSettings_L5('..\param\test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt');

% 验证参数
settings.gpsl5.codeLengthInChips  % 应该是 10230
settings.gpsl5.codeFreqBasis      % 应该是 10.23e6
settings.gpsl5.carrierFreq        % 应该是 1176.45e6
```

### 3. 主程序框架测试
```matlab
% 只有在有实际数据时才能运行
% 否则会在捕获阶段失败（因为doAcquisition_L5还未实现）
```

## 下一步开发路线图

### 优先级 1：核心信号处理
1. ✅ 参数系统
2. ✅ PRN码生成
3. ❌ 捕获算法
4. ❌ 跟踪算法
5. ❌ 帧解码

### 优先级 2：增强功能
- 并行处理支持
- 多相关器跟踪
- 性能优化
- 错误处理

### 优先级 3：验证与测试
- 仿真数据测试
- 实际数据验证
- 性能基准测试
- 与L1信号对比

## 参考文档

### GPS L5
- **IS-GPS-705**: GPS L5 Interface Specification
- **NAVSTAR GPS Space Segment/User Segment L5 Interfaces**

### Galileo E5a
- **Galileo OS SIS ICD**: Galileo Open Service Signal In Space Interface Control Document
- **Galileo E5 Signal Structure**

## 常见问题

### Q: 为什么不直接修改原代码？
A: 为了保持代码库的完整性和稳定性，采用独立模块方式更安全。

### Q: 能否同时处理L1和L5信号？
A: 目前不支持。需要修改主程序逻辑或运行两次（一次L1，一次L5）。

### Q: PRN码初始状态从哪里来？
A: GPS L5从IS-GPS-705获取，Galileo E5a需要从Galileo OS SIS ICD查阅（当前为示例值）。

### Q: 如何验证PRN码正确性？
A: 与标准文档中的测试向量对比，或使用在线PRN码生成器验证。

## 贡献与反馈

如有问题或改进建议，请：
1. 检查Galileo E5a的PRN初始状态值
2. 实现缺失的捕获/跟踪模块
3. 使用实际数据进行测试
4. 报告bug和性能问题

## 版权信息

基于FGI-GSRx v2.0.2
扩展模块: GPS L5 / Galileo E5a Support
许可证: GNU General Public License v3.0
