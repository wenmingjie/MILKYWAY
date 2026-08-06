import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import label, find_objects

def remove_dc_offset(signals, time_us, baseline_region=(0, 1)):
    """
    每个信号去除直流偏置，基于指定的基线时间区域。
    signals: (n, N) 数组
    time_us: (N,) 时间轴 (μs)
    返回: 校正后的 signals
    """
    corrected = []
    for sig in signals:
        mask = (time_us >= baseline_region[0]) & (time_us <= baseline_region[1])
        if np.sum(mask) > 5:
            dc = np.mean(sig[mask])
        else:
            dc = np.mean(sig[:min(10, len(sig))])
        corrected.append(sig - dc)
    return np.array(corrected)

def apply_time_window(signals, time_us, window_us):
    """
    根据时间窗口截取信号。
    window_us: (start, end) 或 (None, None) 表示不截取
    返回: 截取后的 signals, time_us
    """
    if window_us[0] is None or window_us[1] is None:
        return signals, time_us
    mask = (time_us >= window_us[0]) & (time_us <= window_us[1])
    if signals.ndim == 1:
        return signals[mask], time_us[mask]
    else:
        return signals[:, mask], time_us[mask]

def compute_vpp(signal):
    return np.max(signal) - np.min(signal)

def compute_mean_std(signals, axis=0):
    return np.mean(signals, axis=axis), np.std(signals, axis=axis)

def compute_envelope(signal):
    """计算信号的包络（Hilbert变换）"""
    return np.abs(hilbert(signal))

def pulse_duration(time_us, envelope, threshold_dB):
    """
    计算脉冲宽度（以包络低于 threshold_dB 为界）
    返回宽度 (μs)，若无有效区域则返回 None
    """
    peak = np.max(envelope)
    threshold = peak * 10 ** (threshold_dB / 20)
    above = envelope >= threshold
    if not np.any(above):
        return None
    labeled, ncomp = label(above)
    slices = find_objects(labeled)
    max_len = 0
    for sl in slices:
        region = above[sl]
        if np.sum(region) > max_len:
            max_len = np.sum(region)
    dt = time_us[1] - time_us[0]
    return max_len * dt
