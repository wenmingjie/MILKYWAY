import numpy as np
from scipy.fft import fft, fftfreq

def perform_fft_analysis(time_s, voltage, max_freq=None, zero_padding_factor=1):
    """
    同原 fft_analyzer 中函数，但保持一致性。
    返回 dict 含 freqs_valid, fft_complex, fft_norm, fft_dB 等。
    """
    dt = time_s[1] - time_s[0]
    fs = 1.0 / dt
    N = len(voltage)
    sig = voltage - np.mean(voltage)   # 去直流
    # 不加窗（保留原算法）
    N_fft = int(N * zero_padding_factor)
    if N_fft < N:
        raise ValueError("zero_padding_factor 必须 >= 1")
    sig_zp = np.pad(sig, (0, N_fft - N), mode='constant')
    FFT_full = fft(sig_zp)
    freqs_full = fftfreq(N_fft, dt)
    mask_pos = freqs_full >= 0
    freqs = freqs_full[mask_pos]
    FFT = FFT_full[mask_pos]
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        FFT = FFT[mask]
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

def average_spectrum_complex(signals, time_us, max_freq, zero_padding):
    """
    对多条信号的频谱进行复数平均。
    返回同 perform_fft_analysis 的 dict，但为平均结果。
    """
    n_signals = signals.shape[0]
    time_s = time_us * 1e-6
    all_fft_complex = []
    freqs_ref = None
    for i in range(n_signals):
        res = perform_fft_analysis(time_s, signals[i], max_freq, zero_padding)
        if freqs_ref is None:
            freqs_ref = res['freqs_valid']
        all_fft_complex.append(res['fft_complex'])
    fft_complex_avg = np.mean(all_fft_complex, axis=0)
    mag = np.abs(fft_complex_avg)
    mag_norm = mag / (np.max(mag) + 1e-15)
    mag_dB = 20 * np.log10(mag_norm + 1e-15)
    return {
        'freqs_valid': freqs_ref,
        'freqs_valid_mhz': freqs_ref / 1e6,
        'fft_complex': fft_complex_avg,
        'magnitude': mag,
        'fft_norm': mag_norm,
        'fft_dB': mag_dB,
    }

def find_crossing_points(freqs, mag, threshold, peak_idx):
    # 同原 fft_analyzer 中的函数
    left_candidates = np.where(mag[:peak_idx] < threshold)[0]
    if len(left_candidates) > 0:
        left_idx = left_candidates[-1]
    else:
        left_idx = 0
    if left_idx > 0 and left_idx < len(mag) - 1:
        x1, y1 = freqs[left_idx], mag[left_idx]
        x2, y2 = freqs[left_idx + 1], mag[left_idx + 1]
        if y1 != y2:
            f_low = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
        else:
            f_low = x1
    else:
        f_low = freqs[0]
    right_candidates = np.where(mag[peak_idx:] < threshold)[0]
    if len(right_candidates) > 0:
        right_idx = peak_idx + right_candidates[0]
    else:
        right_idx = len(mag) - 1
    if right_idx > 0 and right_idx < len(mag) - 1:
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
    # 同原 fft_analyzer 中的函数，兼容原接口
    if thresholds is None:
        thresholds = {'3dB': 1 / np.sqrt(2)}
    if fmin is not None:
        mask = freqs_valid >= fmin
        freqs_valid = freqs_valid[mask]
        fft_norm = fft_norm[mask]
        fft_dB = fft_dB[mask]
    peak_idx = np.argmax(fft_norm)
    peak_freq_hz = freqs_valid[peak_idx]
    peak_amplitude_dB = fft_dB[peak_idx]
    results = {
        'peak_freq_hz': peak_freq_hz,
        'peak_freq_mhz': peak_freq_hz / 1e6,
        'peak_amplitude_dB': peak_amplitude_dB,
        'peak_idx': peak_idx
    }
    for th_name, th_val in thresholds.items():
        f_low, f_high = find_crossing_points(freqs_valid, fft_norm, th_val, peak_idx)
        bw = f_high - f_low
        center_hz = (f_low + f_high) / 2
        idx_center = np.argmin(np.abs(freqs_valid - center_hz))
        center_dB = fft_dB[idx_center]
        results.update({
            f'{th_name}_low_hz': f_low,
            f'{th_name}_high_hz': f_high,
            f'{th_name}_bw': bw,
            f'{th_name}_center_hz': center_hz,
            f'{th_name}_center_dB': center_dB,
        })
    return results
