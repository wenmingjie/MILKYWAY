import os
import glob
import numpy as np

def load_csv_signals(folder_path, voltage_scale=1000):
    """
    加载文件夹内所有 CSV 文件（格式：第一列索引，第二列时间(s)，第三列电压(V)）
    返回：signals (n, N) 数组（mV），time_us 时间轴 (μs)
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"警告: 在 {folder_path} 中未找到任何 .csv 文件")
        return None, None

    all_signals = []
    ref_time = None
    for fpath in csv_files:
        try:
            data = np.loadtxt(fpath, skiprows=1, delimiter=',')
            if data.ndim != 2 or data.shape[1] < 3:
                continue
            time_s = data[:, 1]
            voltage_V = data[:, 2]
            voltage_mV = voltage_V * voltage_scale
            time_us = time_s * 1e6
            if ref_time is None:
                ref_time = time_us
                all_signals.append(voltage_mV)
            else:
                # 检查时间轴是否一致
                if len(time_us) == len(ref_time) and np.allclose(time_us, ref_time, atol=1e-6):
                    all_signals.append(voltage_mV)
                else:
                    print(f"时间轴不匹配，跳过: {os.path.basename(fpath)}")
        except Exception as e:
            print(f"加载失败 {fpath}: {e}")

    if not all_signals:
        print(f"未加载任何有效信号: {folder_path}")
        return None, None

    return np.array(all_signals), ref_time

def load_csv_average(folder_path, voltage_scale=1000):
    """加载文件夹内所有 CSV 并返回时域平均信号 (mV) 和时间轴 (μs)"""
    signals, time_us = load_csv_signals(folder_path, voltage_scale)
    if signals is None:
        return None, None
    avg_signal = np.mean(signals, axis=0)
    return avg_signal, time_us
