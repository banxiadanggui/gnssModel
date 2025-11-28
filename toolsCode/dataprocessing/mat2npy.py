import numpy as np
from scipy.io import loadmat
import os

############################################
# 你需要修改的部分
mat_path = "data/sharemat/TGD_L1_E1.mat"

output_root = "data/dataset_npy/TGD/"
normal_dir   = output_root + "normal/"
attack_dir   = output_root + "attack/"
fail_dir     = output_root + "tracking_fail/"
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)
os.makedirs(fail_dir, exist_ok=True)

# 时间段设置
t_normal_end = 132   # 正常区间结束
t_attack_start = 138 # 攻击区开始

# tracking 数据点数对应的采样率（需要自动估计）
#mcd 150 158,tgs 133 138,utd 129 133,tgd 132 138
############################################

print("Loading MAT...")
mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

track = mat["trackData"]
channels = track.gpsl1.channel  # 卫星 tracking channel 数组

############################################
# 自动估计 tracking 更新频率（Hz）
# 例如 CN0fromSNR 长度为 377900，总时长约 380 秒 → ~995 Hz
############################################

# 以 CN0 序列长度估计 tracking rate
sample_len = len(channels[0].CN0fromSNR)
total_time = 380  # 秒（你说时间到 380s）
tracking_rate = 1000

print(f"Tracking rate = {tracking_rate:.2f} Hz")

# 将时间转换为 sample index
normal_end_idx  = int(t_normal_end * tracking_rate)
attack_start_idx = int(t_attack_start * tracking_rate)

############################################
# 滑动窗口设置
############################################
window = 2000
stride = 1000

############################################
# 处理每个 channel
############################################

sample_id = 0

for ch in channels:

    # 取我们需要的特征字段
    I_P   = ch.I_P
    Q_P   = ch.Q_P
    dop   = ch.doppler
    cfreq = ch.carrFreq
    cphase = ch.codePhase
    CN0   = ch.CN0fromSNR
    pll   = ch.pllLockIndicator
    fll   = ch.fllLockIndicator
    dll   = ch.dllDiscr

    # 将不同长度对齐为统一长度
    L = min(len(I_P), len(Q_P), len(dop), len(cfreq), len(cphase), 
            len(CN0), len(pll), len(fll), len(dll))

    I_P, Q_P, dop, cfreq, cphase, CN0, pll, fll, dll = \
        I_P[:L], Q_P[:L], dop[:L], cfreq[:L], cphase[:L], \
        CN0[:L], pll[:L], fll[:L], dll[:L]

    # 组合成 (L, F) 的特征矩阵
    features = np.stack([
        I_P, Q_P, dop, cfreq, cphase,
        CN0, pll, fll, dll
    ], axis=1)   # shape (L, 9)

    # 滑动窗口分割
    for start in range(0, L - window, stride):
        end = start + window
        segment = features[start:end]

        # tracking failure 判定（可根据 lockIndicator 设阈值）
        if np.mean(pll[start:end]) < 0.4 or np.mean(fll[start:end]) < 0.4:
            save_path = os.path.join(fail_dir, f"fail_{sample_id:06d}.npy")
            np.save(save_path, segment)
            sample_id += 1
            continue

        # 时间区间标签
        if end <= normal_end_idx:
            save_path = os.path.join(normal_dir, f"normal_{sample_id:06d}.npy")
        elif start >= attack_start_idx:
            save_path = os.path.join(attack_dir, f"attack_{sample_id:06d}.npy")
        else:
            continue  # 跳过跨界片段

        np.save(save_path, segment)
        sample_id += 1

print("Done! NPys saved.")
