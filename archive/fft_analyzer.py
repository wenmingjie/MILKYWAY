import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt

def perform_fft_analysis(
    time_s,
    voltage,
    max_freq=None,
    zero_padding_factor=1
):
    """
    ======================================================================
    通用 FFT 频谱分析（专业版）
    - 不加窗（矩形窗）
    - 不做任何去卷积（H = Y/X）
    - 不做任何正则化
    - 高通滤波在频域进行（可选）
    ======================================================================

    参数:
        time_s : ndarray
            时间轴 (秒)
        voltage : ndarray
            信号电压 (V)
        max_freq : float or None
            最大保留的频率 (Hz)，None = 不限制
        zero_padding_factor : int
            FFT 补零倍数（例如 4 表示补零到原来的 4 倍）
        highpass_freq : float or None
            高通截止频率（Hz）。若提供，则在频域将 |f| < highpass_freq 的分量设为 0。
            注意：此操作在 FFT 后、取正频率前进行，以保证共轭对称性（实信号）。

    返回:
        dict 包含:
            freqs_valid        — 正频率轴 (Hz)
            fft_complex        — 复数频谱 Y(f)（已应用频域高通）
            magnitude          — 幅度 |Y|
            magnitude_norm     — 幅度归一化 (峰值=1)
            magnitude_dB       — 幅度 dB（参考峰值）
            dt, N, fs          — 基本参数
    """

    # ---------------------------------------------------------
    # 基础参数
    # ---------------------------------------------------------
    dt = time_s[1] - time_s[0]
    fs = 1.0 / dt
    N = len(voltage)

    # ---------------------------------------------------------
    # 去直流（强制移除 f=0 成分，等效于 highpass_freq=0+）
    # ---------------------------------------------------------
    sig = voltage - np.mean(voltage)

    # window = np.hanning(len(sig))
    # sig = sig * window
    # ---------------------------------------------------------
    # FFT 补零
    # ---------------------------------------------------------
    N_fft = int(N * zero_padding_factor)
    if N_fft < N:
        raise ValueError("zero_padding_factor 必须 >= 1")

    sig_zp = np.pad(sig, (0, N_fft - N), mode='constant')
    FFT_full = fft(sig_zp)  # 完整 FFT（含负频率）
    freqs_full = fftfreq(N_fft, dt)

    # ---------------------------------------------------------
    # 只保留正频率（包括 f=0）
    # ---------------------------------------------------------
    mask_pos = freqs_full >= 0
    freqs = freqs_full[mask_pos]
    FFT = FFT_full[mask_pos]

    # 限制 max_freq
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        FFT = FFT[mask]

    # ---------------------------------------------------------
    # 计算幅度
    # ---------------------------------------------------------
    mag = np.abs(FFT)
    mag_norm = mag / (np.max(mag) + 1e-12)
    mag_dB = 20 * np.log10(np.maximum(mag_norm, 1e-12))

    return {
        'freqs_valid': freqs,
        'freqs_valid_mhz': freqs / 1e6,
        'fft_complex': FFT,
        'magnitude': mag,
        'fft_norm': mag_norm,
        'fft_dB': mag_dB,
        'dt': dt,
        'N_original': N,
        'N_fft': N_fft,
        'fs': fs,
    }

def find_crossing_points(freqs, mag, threshold, peak_idx):
    """使用线性插值找到精确的交叉点"""
    # 左侧：只在 peak 左边找
    left_candidates = np.where(mag[:peak_idx] < threshold)[0]
    if len(left_candidates) > 0:
        left_idx = left_candidates[-1]
    else:
        left_idx = 0

    if left_idx > 0 and left_idx < len(mag) - 1:
        # 线性插值
        x1, y1 = freqs[left_idx], mag[left_idx]
        x2, y2 = freqs[left_idx + 1], mag[left_idx + 1]
        if y1 != y2:
            f_low = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
        else:
            f_low = x1
    else:
        f_low = freqs[0]

    # 右侧：只在 peak 右边找
    right_candidates = np.where(mag[peak_idx:] < threshold)[0]
    if len(right_candidates) > 0:
        right_idx = peak_idx + right_candidates[0]
    else:
        right_idx = len(mag) - 1

    if right_idx > 0 and right_idx < len(mag) - 1:
        # 线性插值
        x1, y1 = freqs[right_idx - 1], mag[right_idx - 1]
        x2, y2 = freqs[right_idx], mag[right_idx]
        if y1 != y2:
            f_high = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
        else:
            f_high = x2
    else:
        f_high = freqs[-1]

    return f_low, f_high


def calculate_bandwidth(freqs_valid, fft_norm, fft_dB, thresholds=None, fmin=None):
    """
    计算带宽参数（支持自定义阈值）

    参数:
    freqs_valid: 有效频率数组
    fft_norm: 归一化FFT幅度
    fft_dB: FFT幅度(dB)
    thresholds: 自定义阈值字典，例如 {'3dB': 0.707, '6dB': 0.5, '20dB': 0.1}

    返回:
    dict: 包含带宽分析结果的字典
    """
    # 默认阈值
    if thresholds is None:
        thresholds = {
            '3dB': 1 / np.sqrt(2),  # 0.707
            '6dB': 0.5,  # 0.5
            '20dB': 0.1  # 0.1
        }

    # ---------------------------------------------------------
    # 限制分析频率范围（替代高通）
    # ---------------------------------------------------------
    if fmin is not None:
        mask = freqs_valid >= fmin
        freqs_valid = freqs_valid[mask]
        fft_norm = fft_norm[mask]
        fft_dB = fft_dB[mask]

    # 找到峰值频率
    peak_idx = np.argmax(fft_norm)

    # 获取峰值频率和幅度
    peak_freq_hz = freqs_valid[peak_idx]
    peak_freq_mhz = peak_freq_hz / 1e6
    peak_amplitude_dB = fft_dB[peak_idx]

    results = {
        'peak_freq_hz': peak_freq_hz,
        'peak_freq_mhz': peak_freq_mhz,
        'peak_amplitude_dB': peak_amplitude_dB,
        'peak_idx': peak_idx
    }

    # 计算每个阈值的带宽
    for threshold_name, threshold_value in thresholds.items():
        f_low, f_high = find_crossing_points(freqs_valid, fft_norm, threshold_value, peak_idx)
        bw = f_high - f_low
        center_hz = (f_low + f_high) / 2

        # 找到中点对应的幅度值
        def find_amplitude_at_frequency(freqs, mag_dB, target_freq):
            idx = np.argmin(np.abs(freqs - target_freq))
            return mag_dB[idx], idx

        center_dB, center_idx = find_amplitude_at_frequency(freqs_valid, fft_dB, center_hz)

        results.update({
            f'{threshold_name}_low_hz': f_low,
            f'{threshold_name}_high_hz': f_high,
            f'{threshold_name}_bw': bw,
            f'{threshold_name}_center_hz': center_hz,
            f'{threshold_name}_center_dB': center_dB
        })

    return results
