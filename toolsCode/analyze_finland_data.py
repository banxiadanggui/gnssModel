import h5py
import numpy as np
import matplotlib.pyplot as plt

# 分析芬兰数据集结构
mat_files = [
    'gps_mcdfmc_FINAL_SOLUTION.mat',
    'gps_tgdfmc_FINAL_SOLUTION.mat',
    'gps_tgsfmc_FINAL_SOLUTION.mat',
    'gps_utdfmc_FINAL_SOLUTION.mat'
]

data_folder = r'芬兰L1_E1数据集\芬兰L1_E1数据集'

for mat_file in mat_files:
    print(f"\n{'='*70}")
    print(f"分析文件: {mat_file}")
    print('='*70)

    file_path = f"{data_folder}/{mat_file}"

    with h5py.File(file_path, 'r') as f:
        # 检查navData长度
        if 'navData' in f:
            nav_data = f['navData']
            num_epochs = nav_data.shape[0]
            print(f"\nnavData epochs数量: {num_epochs}")

        # 检查settings结构
        if 'settings' in f:
            settings = f['settings']
            if 'nav' in settings:
                nav_settings = settings['nav']
                if 'navSolPeriod' in nav_settings:
                    sample_rate = nav_settings['navSolPeriod'][0][0]
                    print(f"导航解算周期 (navSolPeriod): {sample_rate} ms")
                    total_time = num_epochs * sample_rate / 1000
                    print(f"总时长: {total_time:.2f} 秒")

        # 检查trackData结构
        if 'trackData' in f:
            track_data = f['trackData']
            if 'gpsl1' in track_data:
                gpsl1 = track_data['gpsl1']

                # 获取通道数 - 简化处理
                if 'nrObs' in gpsl1:
                    try:
                        nr_obs = gpsl1['nrObs'][()]
                        if hasattr(nr_obs, '__len__'):
                            nr_obs = nr_obs[0][0]
                        print(f"跟踪通道数 (nrObs): {nr_obs}")
                    except:
                        print("无法读取nrObs")

                # 检查channel结构
                if 'channel' in gpsl1:
                    print(f"trackData.gpsl1.channel存在")

print("\n" + "="*70)
print("分析完成！")
print("="*70)
