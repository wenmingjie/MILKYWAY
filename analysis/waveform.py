import numpy as np

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
