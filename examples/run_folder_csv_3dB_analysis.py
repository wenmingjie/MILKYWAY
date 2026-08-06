"""
双文件夹CSV时频分析（重构版）
功能：
- 从两个文件夹分别加载CSV（频域/时域）
- 计算时域参数、复数平均频谱、带宽、HD2
- 绘制双轴叠加图（含信息框）
"""
import os
import numpy as np
from io.csv_io import load_csv_signals, load_csv_average
from analysis.waveform import remove_dc_offset, apply_time_window, compute_envelope, pulse_duration
from analysis.spectrum import average_spectrum_complex, calculate_bandwidth
from plot.figure import plot_time_freq_dual_axis_with_boxes
from plot.style import set_style

# ==================== 配置参数 ====================
FOLDER_1CYCLE = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz' # 频域
FOLDER_3CYCLE = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz' # 时域
DIE_ID = 'STB3SP11W00_R3C0_L' 

WINDOW_1CYCLE = (6.5, 8.0)   # 频域窗口 (μs)
WINDOW_3CYCLE = (6.5, 8.0)   # 时域窗口 (μs)

FFT_MAX_FREQ = 30e6
FFT_ZERO_PADDING = 4

OUTPUT_DIR = r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 主程序 ====================
def main():
    set_style()
    print("="*60)
    print("双文件夹CSV时频分析（重构版）")
    print("="*60)

    # 1. 加载数据
    signals1, time1 = load_csv_signals(FOLDER_1CYCLE)
    sig3_avg, time3 = load_csv_average(FOLDER_3CYCLE)
    if signals1 is None or sig3_avg is None:
        return

    # 2. 去直流
    sig3_avg = remove_dc_offset(sig3_avg, time3)

    # 3. 应用窗口
    # 频域：所有信号分别截窗
    signals1_win = []
    time1_win = None
    for sig in signals1:
        sig_win, t_win = apply_time_window(sig, time1, WINDOW_1CYCLE)
        signals1_win.append(sig_win)
        if time1_win is None:
            time1_win = t_win
    signals1_win = np.array(signals1_win)

    # 时域：平均信号截窗
    sig3_win, time3_win = apply_time_window(sig3_avg, time3, WINDOW_3CYCLE)

    # 4. 时域参数
    vpp = np.max(sig3_win) - np.min(sig3_win)
    peak_voltage = np.max(np.abs(sig3_win))
    env = compute_envelope(sig3_win)
    dur_6dB = pulse_duration(time3_win, env, -6)
    dur_20dB = pulse_duration(time3_win, env, -20)
    dur6_str = "N/A" if dur_6dB is None else f"{dur_6dB:.2f}"
    dur20_str = "N/A" if dur_20dB is None else f"{dur_20dB:.2f}"
    time_info = (f"Vpp : {vpp:.2f} mV\n"
                 f"Peak : {peak_voltage:.2f} mV\n"
                 f"-6 dB Width : {dur6_str} μs\n"
                 f"-20 dB Width : {dur20_str} μs")

    # 5. 频域参数（复数平均）
    fft_avg = average_spectrum_complex(signals1_win, time1_win, FFT_MAX_FREQ, FFT_ZERO_PADDING)
    bw_results = calculate_bandwidth(
        fft_avg['freqs_valid'],
        fft_avg['fft_norm'],
        fft_avg['fft_dB'],
        thresholds={'3dB': 1/np.sqrt(2)},
        fmin=None
    )
    peak_freq_mhz = bw_results['peak_freq_mhz']
    peak_dB = bw_results['peak_amplitude_dB']
    center_hz = bw_results.get('3dB_center_hz')
    bw_hz = bw_results.get('3dB_bw')
    rel_bw = (bw_hz / center_hz * 100) if center_hz and center_hz > 0 else 0

    # 计算 HD2
    hd2_dBc = None
    hd2_freq = 2 * bw_results['peak_freq_hz']
    if hd2_freq <= FFT_MAX_FREQ:
        hd2_dB = np.interp(hd2_freq, fft_avg['freqs_valid'], fft_avg['fft_dB'])
        hd2_dBc = hd2_dB - peak_dB

    # 构建频域信息文本
    freq_info = (f"Peak Freq. : {peak_freq_mhz:.2f} MHz\n"
                 f"Center Freq. : {center_hz/1e6:.2f} MHz\n" if center_hz else ""
                 f"-3dB Low : {bw_results['3dB_low_hz']/1e6:.2f} MHz\n" if '3dB_low_hz' in bw_results else ""
                 f"-3dB High : {bw_results['3dB_high_hz']/1e6:.2f} MHz\n" if '3dB_high_hz' in bw_results else ""
                 f"-3dB BW : {bw_hz/1e6:.2f} MHz\n"
                 f"Rel. BW : {rel_bw:.1f} %\n"
                 f"HD2 : {hd2_dBc:.1f} dBc" if hd2_dBc is not None else "HD2 : N/A")

    # 6. 绘图
    title = f"Transmit Response Waveform of Device {DIE_ID}"
    save_path = os.path.join(OUTPUT_DIR, f"Transmit_Response_Waveform_{DIE_ID}_avg_spectrum.png")
    fig, ax_time, ax_freq = plot_time_freq_dual_axis_with_boxes(
        time_us=time3_win,
        signal=sig3_win,
        freqs_mhz=fft_avg['freqs_valid_mhz'],
        spectrum_dB=fft_avg['fft_dB'],
        peak_freq_mhz=peak_freq_mhz,
        peak_dB=peak_dB,
        bandwidth_results=bw_results,
        hd2_dBc=hd2_dBc,
        title=title,
        time_info_text=time_info,
        freq_info_text=freq_info,
        freq_xlim=(0, FFT_MAX_FREQ/1e6),
        freq_ylim=(-60, 5),
        save_path=save_path
    )
    plt.show()

    # 打印摘要
    print("\n【频域参数】")
    print(f"  信号数量 = {signals1_win.shape[0]}")
    print(f"  Peak freq. = {peak_freq_mhz:.3f} MHz")
    print(f"  Center freq. = {center_hz/1e6:.3f} MHz" if center_hz else "")
    print(f"  -3dB BW = {bw_hz/1e6:.3f} MHz" if bw_hz else "")
    print(f"  Rel. BW = {rel_bw:.2f}%")
    if hd2_dBc is not None:
        print(f"  HD2 = {hd2_dBc:.1f} dBc")
    print("="*60)

if __name__ == "__main__":
    main()
