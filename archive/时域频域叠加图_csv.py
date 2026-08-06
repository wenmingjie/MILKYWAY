"""
双文件夹CSV时频分析 - 复数平均频谱（专业FFT版）
=====================================================
功能：
1. 从 FOLDER_1CYCLE 加载所有CSV文件，分别做FFT，复数平均得到平均频谱
2. 从 FOLDER_3CYCLE 加载所有CSV文件，取时域平均信号用于时域显示
3. 计算时域参数（Vpp、峰值、-6/-20dB宽度）
4. 计算频域参数（峰值、-3dB中心/带宽/高低频点，线性插值）
5. 绘制双轴叠加图（时域左下，频域右上），图例置于右上角
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import hilbert
from scipy.ndimage import label, find_objects

# ==================== 配置参数 ====================
FOLDER_1CYCLE = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz'   # 用于频域（多条信号频谱平均）
FOLDER_3CYCLE = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz'   # 用于时域（时域平均）
die_id = 'STB3SP11W00_R3C0_L'

# WINDOW_1CYCLE = (-0.5, 1.0)   # 频域信号窗口 (μs)
# WINDOW_3CYCLE = (-0.5, 1.0)   # 时域信号窗口 (μs)
WINDOW_1CYCLE = (6.5, 8.0)   # 频域信号窗口 (μs)
WINDOW_3CYCLE = (6.5, 8.0)   # 时域信号窗口 (μs)

FFT_MAX_FREQ = 30e6          # 最高分析频率 (Hz)
FFT_ZERO_PADDING = 4         # 零填充倍数

OUTPUT_DIR = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 工具函数 ====================
def load_csv_all(folder_path):
    """加载文件夹内所有CSV文件，返回信号数组 (n_signals, n_points) 和时间轴 (μs)"""
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
            voltage_mV = voltage_V * 1000
            time_us = time_s * 1e6
            if ref_time is None:
                ref_time = time_us
                all_signals.append(voltage_mV)
            else:
                if len(time_us) == len(ref_time) and np.allclose(time_us, ref_time, atol=1e-6):
                    all_signals.append(voltage_mV)
                else:
                    print(f"时间轴不匹配，跳过: {os.path.basename(fpath)}")
        except Exception as e:
            print(f"加载失败 {fpath}: {e}")

    if not all_signals:
        print(f"未加载任何有效信号: {folder_path}")
        return None, None

    signals = np.array(all_signals)   # shape: (n, points)
    return signals, ref_time

def load_csv_average(folder_path):
    """加载所有CSV并返回时域平均信号 (mV) 和时间轴 (μs)"""
    signals, time_us = load_csv_all(folder_path)
    if signals is None:
        return None, None
    avg_signal = np.mean(signals, axis=0)
    return avg_signal, time_us

def remove_dc_offset(signal, time_us, baseline_region=(0, 1)):
    mask = (time_us >= baseline_region[0]) & (time_us <= baseline_region[1])
    if np.sum(mask) > 5:
        dc = np.mean(signal[mask])
    else:
        dc = np.mean(signal[:min(10, len(signal))])
    return signal - dc

def apply_time_window(signal, time_us, window_us):
    if window_us[0] is None or window_us[1] is None:
        return signal, time_us
    mask = (time_us >= window_us[0]) & (time_us <= window_us[1])
    return signal[mask], time_us[mask]

# ==================== 专业 FFT 算法（来自第一个代码） ====================
def perform_fft_analysis(
    time_s,
    voltage,
    max_freq=None,
    zero_padding_factor=1
):
    """
    通用 FFT 频谱分析（专业版）
    - 不加窗（矩形窗）
    - 去直流
    - 零填充
    """
    dt = time_s[1] - time_s[0]
    fs = 1.0 / dt
    N = len(voltage)

    # 去直流
    sig = voltage - np.mean(voltage)

    window = np.hanning(len(sig))
    sig = sig * window

    # 零填充
    N_fft = int(N * zero_padding_factor)
    if N_fft < N:
        raise ValueError("zero_padding_factor 必须 >= 1")

    sig_zp = np.pad(sig, (0, N_fft - N), mode='constant')
    FFT_full = np.fft.fft(sig_zp)
    freqs_full = np.fft.fftfreq(N_fft, dt)

    # 只保留正频率（含 f=0）
    mask_pos = freqs_full >= 0
    freqs = freqs_full[mask_pos]
    FFT = FFT_full[mask_pos]

    # 限制 max_freq
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

def find_crossing_points(freqs, mag, threshold, peak_idx):
    """使用线性插值找到精确的交叉点"""
    # 左侧
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

    # 右侧
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
    """
    计算带宽参数（支持自定义阈值）
    """
    if thresholds is None:
        thresholds = {
            '3dB': 1 / np.sqrt(2),   # 0.707
            '6dB': 0.5,
            '20dB': 0.1
        }

    if fmin is not None:
        mask = freqs_valid >= fmin
        freqs_valid = freqs_valid[mask]
        fft_norm = fft_norm[mask]
        fft_dB = fft_dB[mask]

    peak_idx = np.argmax(fft_norm)
    peak_freq_hz = freqs_valid[peak_idx]
    peak_freq_mhz = peak_freq_hz / 1e6
    peak_amplitude_dB = fft_dB[peak_idx]

    results = {
        'peak_freq_hz': peak_freq_hz,
        'peak_freq_mhz': peak_freq_mhz,
        'peak_amplitude_dB': peak_amplitude_dB,
        'peak_idx': peak_idx
    }

    for threshold_name, threshold_value in thresholds.items():
        f_low, f_high = find_crossing_points(freqs_valid, fft_norm, threshold_value, peak_idx)
        bw = f_high - f_low
        center_hz = (f_low + f_high) / 2

        # 中点对应的幅度
        idx_center = np.argmin(np.abs(freqs_valid - center_hz))
        center_dB = fft_dB[idx_center]

        results.update({
            f'{threshold_name}_low_hz': f_low,
            f'{threshold_name}_high_hz': f_high,
            f'{threshold_name}_bw': bw,
            f'{threshold_name}_center_hz': center_hz,
            f'{threshold_name}_center_dB': center_dB,
        })

    return results

def average_spectrum(signals, time_us, max_freq, zero_padding):
    """
    对多个信号分别做FFT，然后复数平均，返回平均频谱结果（与 perform_fft_analysis 结构相同）
    """
    n_signals = signals.shape[0]
    all_fft_complex = []
    freqs_ref = None

    # 转换为秒
    time_s = time_us * 1e-6

    for i in range(n_signals):
        fft_result = perform_fft_analysis(
            time_s=time_s,
            voltage=signals[i],
            max_freq=max_freq,
            zero_padding_factor=zero_padding
        )
        if freqs_ref is None:
            freqs_ref = fft_result['freqs_valid']
        all_fft_complex.append(fft_result['fft_complex'])

    # 复数平均
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

def compute_envelope(signal):
    return np.abs(hilbert(signal))

def pulse_duration(time_us, envelope, threshold_dB):
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

# ==================== 主程序 ====================
def main():
    print("="*60)
    print("双文件夹CSV时频分析（复数平均频谱）")
    print("="*60)

    # 1. 加载频域数据（所有信号）
    signals1, time1 = load_csv_all(FOLDER_1CYCLE)
    if signals1 is None:
        print("频域数据加载失败。")
        return

    # 2. 加载时域数据（平均信号）
    sig3_avg, time3 = load_csv_average(FOLDER_3CYCLE)
    if sig3_avg is None:
        print("时域数据加载失败。")
        return

    # 3. 去直流偏置
    # 频域信号每条分别去直流（在 perform_fft_analysis 中已做）
    sig3_avg = remove_dc_offset(sig3_avg, time3)

    # 4. 截取窗口
    # 频域窗口：对每条信号截窗
    signals1_win = []
    time1_win = None
    for sig in signals1:
        sig_win, t_win = apply_time_window(sig, time1, WINDOW_1CYCLE)
        signals1_win.append(sig_win)
        if time1_win is None:
            time1_win = t_win
    signals1_win = np.array(signals1_win)

    # 时域窗口
    sig3_win, time3_win = apply_time_window(sig3_avg, time3, WINDOW_3CYCLE)

    # 5. 时域参数
    vpp = np.max(sig3_win) - np.min(sig3_win)
    peak_voltage = np.max(np.abs(sig3_win))
    peak_idx_time = np.argmax(np.abs(sig3_win))
    peak_time = time3_win[peak_idx_time]
    peak_val = sig3_win[peak_idx_time]

    env3 = compute_envelope(sig3_win)
    dur_6dB = pulse_duration(time3_win, env3, -6)
    dur_20dB = pulse_duration(time3_win, env3, -20)

    # 6. 频域参数（复数平均频谱）
    fft_avg = average_spectrum(signals1_win, time1_win, FFT_MAX_FREQ, FFT_ZERO_PADDING)

    # 计算带宽（只取 -3dB）
    bandwidth = calculate_bandwidth(
        freqs_valid=fft_avg['freqs_valid'],
        fft_norm=fft_avg['fft_norm'],
        fft_dB=fft_avg['fft_dB'],
        thresholds={'3dB': 1/np.sqrt(2)},
        fmin=None
    )

    # 提取参数
    peak_freq = bandwidth['peak_freq_hz']
    peak_dB = bandwidth['peak_amplitude_dB']
    low_freq = bandwidth['3dB_low_hz']
    high_freq = bandwidth['3dB_high_hz']
    center_freq = bandwidth['3dB_center_hz']
    bw = bandwidth['3dB_bw']
    rel_bw = (bw / center_freq) * 100 if center_freq > 0 else 0

    # ----- 计算 HD2 -----
    hd2_freq = 2 * peak_freq  # 二次谐波频率 (Hz)
    if hd2_freq <= FFT_MAX_FREQ:
        hd2_dB = np.interp(hd2_freq, fft_avg['freqs_valid'], fft_avg['fft_dB'])
        hd2_rel_dB = hd2_dB - peak_dB  # 相对于基频的 dBc
        # hd2_str = f"HD2 : {hd2_freq / 1e6:.2f} MHz ({hd2_rel_dB:.1f} dBc)"
    else:
        # hd2_str = "HD2 : N/A (超出分析范围)"
        print("HD2 : N/A (超出分析范围)")

    # 打印结果
    print("\n【频域参数（多次测量复数平均）】")
    print(f"  信号数量 = {signals1_win.shape[0]}")
    print(f"  Peak freq. = {peak_freq/1e6:.3f} MHz")
    print(f"  Center freq. = {center_freq/1e6:.3f} MHz")
    print(f"  -3dB Low = {low_freq/1e6:.3f} MHz")
    print(f"  -3dB High = {high_freq/1e6:.3f} MHz")
    print(f"  -3dB BW = {bw/1e6:.3f} MHz")
    print(f"  Rel. BW = {rel_bw:.2f}%")

    # =================== 绘图 ===================
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Transmit Response Waveform of Device {die_id}", fontsize=16, fontweight='bold', y=0.98)
    # fig.suptitle(f"Electrical Excitation Waveform of Device {die_id}", fontsize=16, fontweight='bold', y=0.98)

    # Time axis (左下)
    ax_time = fig.add_axes([0.10, 0.12, 0.68, 0.76])
    ax_time.set_xlim(time3_win[0], time3_win[-1])
    ax_time.set_xlabel("Time (μs)", fontsize=13.5, color="#0072BD")
    ax_time.set_ylabel("Voltage (mV)", fontsize=13.5, color="#0072BD")
    ax_time.tick_params(axis='both', direction='in', length=5, width=1.2, labelsize=11)
    ax_time.tick_params(axis='y', colors="#0072BD")
    ax_time.tick_params(axis='x', colors="#0072BD")
    for s in ax_time.spines.values():
        s.set_linewidth(1.2)
    ax_time.spines['left'].set_color("#0072BD")
    ax_time.spines['bottom'].set_color("#0072BD")
    ax_time.spines['top'].set_color("#D95319")
    ax_time.spines['right'].set_color("#D95319")
    ax_time.grid(True, linestyle='--', linewidth=0.5, alpha=0.25)

    # Frequency axis (右上)
    ax_freq = fig.add_axes(ax_time.get_position(), frameon=False)
    ax_freq.set_xlim(0, FFT_MAX_FREQ / 1e6)
    ax_freq.set_ylim(-60, 5)
    ax_freq.xaxis.tick_top()
    ax_freq.xaxis.set_label_position('top')
    ax_freq.yaxis.tick_right()
    ax_freq.yaxis.set_label_position('right')
    ax_freq.set_xlabel("Frequency (MHz)", fontsize=13.5, color="#D95319")
    ax_freq.set_ylabel("Normalized Magnitude (dB)", fontsize=13.5, color="#D95319")
    ax_freq.tick_params(axis='x', direction='in', length=5, width=1.2, labelsize=11, color='#D95319', labelcolor='#D95319')
    ax_freq.tick_params(axis='y', direction='in', length=5, width=1.2, labelsize=11, color='#D95319', labelcolor='#D95319')
    ax_freq.spines['top'].set_visible(True)
    ax_freq.spines['right'].set_visible(True)
    ax_freq.spines['top'].set_color("#D95319")
    ax_freq.spines['right'].set_color("#D95319")
    ax_freq.spines['top'].set_linewidth(1.2)
    ax_freq.spines['right'].set_linewidth(1.2)
    ax_freq.spines['left'].set_visible(False)
    ax_freq.spines['bottom'].set_visible(False)

    ax_freq.set_zorder(1)
    ax_time.set_zorder(2)
    ax_time.patch.set_alpha(0)

    # ---- 绘制频域（复数平均频谱） ----
    ax_freq.plot(fft_avg['freqs_valid_mhz'], fft_avg['fft_dB'], color="#D95319", lw=2.2, zorder=1)
    ax_freq.scatter(peak_freq / 1e6, peak_dB, color='red', s=45, zorder=3)
    center_dB = np.interp(center_freq, fft_avg['freqs_valid'], fft_avg['fft_dB'])
    ax_freq.scatter(center_freq / 1e6, center_dB, color='orange', s=45, zorder=3)
    ax_freq.axhline(peak_dB - 3, color='#D95319', ls=':', lw=1.2)
    ax_freq.axvline(low_freq / 1e6, color='#D95319', ls='--', lw=1.0)
    ax_freq.axvline(high_freq / 1e6, color='#D95319', ls='--', lw=1.0)

    # ---- 绘制时域 ----
    ax_time.plot(time3_win, sig3_win, color="#0072BD", lw=2.5, zorder=100)
    ax_time.scatter(peak_time, peak_val, color='red', s=45, zorder=101)

    # ---- 图例信息 ----
    dur6 = "N/A" if dur_6dB is None else f"{dur_6dB:.2f}"
    dur20 = "N/A" if dur_20dB is None else f"{dur_20dB:.2f}"
    time_info = (
        f"Vpp : {vpp:.2f} mV\n"
        f"Peak : {peak_voltage:.2f} mV\n"
        f"-6 dB Width : {dur6} μs\n"
        f"-20 dB Width : {dur20} μs"
    )
    freq_info = (
        f"Peak Freq. : {peak_freq / 1e6:.2f} MHz\n"
        f"Center Freq. : {center_freq / 1e6:.2f} MHz\n"
        f"-3dB Low Freq. : {low_freq / 1e6:.2f} MHz\n"
        f"-3dB High Freq. : {high_freq / 1e6:.2f} MHz\n"
        f"-3 dB BW : {bw / 1e6:.2f} MHz\n"
        f"-3 dB Rel. BW : {rel_bw:.1f} %\n"
        f"HD2 : {hd2_rel_dB:.1f} dBc"
    )
    bbox_style = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=1.0, alpha=0.5)
    ax_time.text(0.985, 0.98, time_info, transform=ax_time.transAxes, ha='right', va='top',
                 fontsize=10, family='Arial', bbox=bbox_style, zorder=200)
    ax_time.text(0.985, 0.78, freq_info, transform=ax_time.transAxes, ha='right', va='top',
                 fontsize=10, family='Arial', bbox=bbox_style, zorder=200)

    # ---- 保存 ----
    # save_path = os.path.join(OUTPUT_DIR, f"Excitation_Waveform_{die_id}_avg_spectrum.png")
    save_path = os.path.join(OUTPUT_DIR, f"Transmit_Response_Waveform_{die_id}_avg_spectrum.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to:\n{save_path}")
    plt.show()

    print("="*60)
    print("分析完成。")

if __name__ == "__main__":
    main()
