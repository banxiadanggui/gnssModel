# 重要参数缺失报告

## 📊 实际字段检查结果

### ✅ 已导出的参数

#### Python程序导出 (UTD_statResults.csv - 62个字段)
来自 **statResults**：
- 位置、速度、误差统计、精度评估等高层次定位结果

#### MATLAB程序导出 (Data_export.csv - 11个字段)
来自 **obsData → channel**：
- ✓ carrierFreq (载波频率)
- ✓ corrP (校正后伪距)
- ✓ trueRange (真实距离)
- ✓ rangeResid (距离残差)
- ✓ doppler (多普勒)
- ✓ dopplerResid (多普勒残差)
- ✓ tow (周内时)
- ✓ transmitTime (发射时间)
- ✓ satId (卫星ID)
- ✓ receiverTow (接收机周内时)
- ✓ signalName (信号名称)

---

## ⚠️ 重要但未导出的参数

### 🔴 优先级1 - 关键缺失（在obsData.channel中存在但未导出）

1. **SNR** (信噪比)
   - 📍 位置: obsData → channel → SNR
   - 🎯 用途: 信号质量评估、干扰检测的核心指标
   - ⚠️ **这是最重要的缺失参数！**

2. **clockCorr** (钟差改正)
   - 📍 位置: obsData → channel → clockCorr
   - 🎯 用途: 卫星钟差改正值，精密定位必需

3. **ionoCorr** (电离层改正)
   - 📍 位置: obsData → channel → ionoCorr
   - 🎯 用途: 电离层延迟改正，定位精度分析

4. **tropoCorr** (对流层改正)
   - 📍 位置: obsData → channel → tropoCorr
   - 🎯 用途: 对流层延迟改正

5. **rawP** (原始伪距)
   - 📍 位置: obsData → channel → rawP
   - 🎯 用途: 未校正的原始伪距观测值

6. **codephase** (码相位)
   - 📍 位置: obsData → channel → codephase
   - 🎯 用途: 码相位测量，定位基础观测量

7. **week** (GPS周)
   - 📍 位置: obsData → channel → week
   - 🎯 用途: 时间标识

### 🟡 优先级2 - 重要辅助参数（在obsData.channel中存在）

8. **bObsOk** (观测有效标志)
   - 📍 位置: obsData → channel → bObsOk
   - 🎯 用途: 判断观测值是否有效

9. **bEphOk** (星历有效标志)
   - 📍 位置: obsData → channel → bEphOk
   - 🎯 用途: 判断星历是否有效

10. **bParityOk** (校验位正确标志)
    - 📍 位置: obsData → channel → bParityOk
    - 🎯 用途: 数据完整性检查

11. **bPreambleOk** (前导码检测标志)
    - 📍 位置: obsData → channel → bPreambleOk
    - 🎯 用途: 帧同步检测

### 🟢 优先级3 - satData和navData中的重要信息

#### 从 **satData** (370个历元)
- 包含 `gpsl1` 和 `gale1b` 两个系统的卫星数据
- 📍 需要进一步检查是否包含：
  - 卫星高度角 (elevation)
  - 卫星方位角 (azimuth)
  - 卫星位置 (satPos)
  - 卫星速度 (satVel)

#### 从 **navData** (370个历元)
- ✓ 已知字段：Pos (位置), Vel (速度), Time (时间)
- 📍 需要检查是否包含：
  - DOP值 (GDOP, PDOP, HDOP, VDOP, TDOP)
  - 使用的卫星数量 (nrSats)
  - 使用的卫星列表 (usedSats)

### ⚪ 不存在的数据

#### trackData
- ❌ 该MAT文件中不存在 trackData 变量
- 原本期望包含：CN0 (载噪比)、载波相位、锁相环输出等
- **注意：SNR在obsData中存在，可以作为替代**

---

## 📈 影响分析

### 对于干扰检测和信号质量分析：

| 参数 | 重要性 | 状态 | 影响 |
|------|--------|------|------|
| **SNR/CN0** | 🔴 极高 | ✗ 未导出 | 无法直接评估信号质量和检测干扰 |
| 载波相位 | 🔴 高 | ✗ 不存在 | 无法进行高精度相位分析 |
| 原始伪距 | 🟡 中 | ✗ 未导出 | 无法分析原始观测值 |
| 改正项 | 🟡 中 | ✗ 未导出 | 无法分析各项误差改正 |
| 有效性标志 | 🟢 低-中 | ✗ 未导出 | 无法判断数据质量 |

---

## 🎯 建议优先补充导出的参数（按重要性排序）

### 立即补充（关键）：
1. **SNR** - 信号质量核心指标
2. **clockCorr, ionoCorr, tropoCorr** - 误差改正项
3. **rawP** - 原始伪距
4. **week** - 完整时间标识

### 建议补充（重要）：
5. **codephase** - 码相位
6. **bObsOk, bEphOk** - 数据有效性标志

### 可选补充（有用）：
7. **satData 中的卫星几何信息**（如果存在）
8. **navData 中的 DOP 值**（如果存在）

---

## 💡 总结

**当前导出情况：**
- ✅ Python: 定位结果和统计 (62个字段)
- ✅ MATLAB: 基础观测值 (11个字段)
- ❌ 缺失: 信号质量指标、误差改正、数据质量标志

**关键发现：**
- ⚠️ **SNR 字段存在但未导出** - 这是信号质量分析的核心指标
- ⚠️ 多个重要改正项（钟差、电离层、对流层）未导出
- ⚠️ 数据质量标志位未导出
- ℹ️ trackData 不存在（可能该数据集未包含底层跟踪信息）

**建议行动：**
扩展 MATLAB 脚本，至少补充导出 SNR、改正项和原始伪距，以支持完整的信号质量和干扰分析。
