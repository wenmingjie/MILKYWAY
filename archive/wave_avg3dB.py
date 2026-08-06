"""
超声波系统响应高精度分析（简化版 - 单文件模式）
==============================================
功能：
1. 单波形响应加载：Pickle格式响应
2. 时域处理：多信号平均、偏置校正、时域窗口提取
3. 频域平均：先平均响应频谱
4. 频域分析：单次响应频谱、平均响应频谱
5. 可视化：时域波形、时域窗口波形、频谱叠加、平均频谱

特点：只处理响应信号，不进行反卷积
输入：单个 Pickle 响应文件（包含多次重复测量）
输出：响应频谱分析图表
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from fft_analyzer import perform_fft_analysis, calculate_bandwidth
from save_fig_utils import save_fig_all_formats

# ==================== 配置参数 =====================
# BASE_DIR = r'Z:\Measurements\2026\top_metal_under_cavity_E_Tx_20260415\1cycle\R3C3-R3C4'
# file_name = r'MKW_2ADDF068SHC6_R3C4_E_f20.6MHz_TFullMetalpulse_BP60V_1cycles_d10mm_11times_square_mwen_20260415.pickle'
# die_id = '2ADDF068SHC6_R3C4_E'
# PICKLE_PATH = os.path.join(BASE_DIR, file_name)
# # OUTPUT_DIR = os.path.join(BASE_DIR, f'analysis_1cycle_NoMetal-DCef60V_{die_id}')
# # OUTPUT_DIR = os.path.join(BASE_DIR, f'analysis_1cycle_HalfMetal-DCef20V_{die_id}')
# OUTPUT_DIR = os.path.join(BASE_DIR, f'analysis_1cycles_{die_id}-1')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = r'Y:\Measurements\2026\20260722'
file_name = r'MKW_STB3SP11W00_R3C0_L_f10.0MHz_Txallchannelpulse_HV20Vpp_1cycles_d5mm_11times_gaus_T24.8C_zshu_20260723.pickle'
die_id = 'STB3SP11W00_R3C0_L'
PICKLE_PATH = os.path.join(BASE_DIR, file_name)
OUTPUT_DIR = os.path.join(BASE_DIR, f'analysis_{die_id}_guas-5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_FORMATS = ['png']

# 窗口设置
USE_RESPONSE_WINDOW = True
RESPONSE_WINDOW_START = 3.3
RESPONSE_WINDOW_END = 4.3
# RESPONSE_WINDOW_END = 8.5
# RESPONSE_WINDOW_START = 3
# RESPONSE_WINDOW_END = 6

# FFT参数
FFT_MAX_FREQ = 20e6
FFT_ZERO_PADDING = 4

# ==================== 工具函数 =====================
def load_single_response_file(file_path):
    """加载单个响应Pickle文件，返回所有波形及时间轴"""
    try:
        data = pd.read_pickle(file_path)
        waveforms_df = data['waveforms']
        # waveforms_df = data
        n_tests = len(waveforms_df)
        print(f"加载 {n_tests} 条波形: {os.path.basename(file_path)}")

        time_vector = np.array(waveforms_df['time'].iloc[0])  # 单位：秒
        time_us = time_vector * 1e6

        signals_V = np.vstack([np.array(row['ch2']) for _, row in waveforms_df.iterrows()])
        signals_mV = signals_V * 1000  # V → mV

        print(f"参考时间轴: {len(time_us)} 点")
        return signals_mV, time_us

    except Exception as e:
        print(f"错误: 处理响应文件 {file_path} 时出错: {e}")
        return None, None

def remove_dc_offset_individual(signals, time_array, baseline_region=(0, 1)):
    """去除每个信号的直流偏置"""
    corrected_signals = []
    for signal in signals:
        baseline_mask = (time_array >= baseline_region[0]) & (time_array <= baseline_region[1])
        if np.sum(baseline_mask) > 5:
            dc_offset = np.mean(signal[baseline_mask])
        else:
            baseline_end_idx = max(10, len(signal) // 10)
            dc_offset = np.mean(signal[:baseline_end_idx])
        corrected_signals.append(signal - dc_offset)
    return np.array(corrected_signals)

def average_response_fft(all_resp_win, time_resp_win,
                          max_freq=30e6, zero_padding=8):
    """
    对所有响应做 FFT，然后在频域进行复数平均
    """
    all_fft_complex = []
    freqs_ref = None

    print(f"\n开始对 {all_resp_win.shape[0]} 条响应做 FFT 并进行频谱平均...")

    for i in range(all_resp_win.shape[0]):
        fft_single = perform_fft_analysis(
            time_resp_win,
            all_resp_win[i],
            max_freq=max_freq,
            zero_padding_factor=zero_padding
        )

        if freqs_ref is None:
            freqs_ref = fft_single['freqs_valid']

        all_fft_complex.append(fft_single['fft_complex'])

    all_fft_complex = np.array(all_fft_complex)

    # 复数平均
    fft_complex_avg = np.mean(all_fft_complex, axis=0)

    mag = np.abs(fft_complex_avg)
    mag_norm = mag / (np.max(mag) + 1e-15)
    mag_dB = 20 * np.log10(mag_norm + 1e-15)

    return {
        'freqs_valid': freqs_ref,
        'freqs_valid_mhz': freqs_ref / 1e6,
        'fft_complex': fft_complex_avg,
        'fft_mag': mag,
        'fft_norm': mag_norm,
        'fft_dB': mag_dB
    }

def save_results_to_csv(results, output_dir, filename="response_analysis_results.csv"):
    """保存分析结果到CSV文件"""
    import csv
    from datetime import datetime

    csv_path = os.path.join(output_dir, filename)

    # 准备要保存的数据
    data_to_save = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'die_id': die_id,
        'response_files_count': results.get('response_files_count', 0),
        'response_window_start_us': RESPONSE_WINDOW_START,
        'response_window_end_us': RESPONSE_WINDOW_END,
        'response_window_vpp_mV': results.get('response_window_vpp', 0),
        'fft_max_freq_MHz': FFT_MAX_FREQ / 1e6,
    }

    # 添加响应频谱结果
    if 'response_spectrum' in results:
        resp = results['response_spectrum']
        data_to_save.update({
            'response_peak_freq_MHz': resp.get('peak_freq_MHz', 0),
            'response_peak_mag_dB': resp.get('peak_mag_dB', 0),
            'response_3dB_bw_MHz': resp.get('3dB_bw_MHz', 0),
            'response_3dB_center_freq_MHz': resp.get('3dB_center_freq_MHz', 0),
            'response_relative_bw_percent': resp.get('relative_bw_percent', 0),
        })

    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())

        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()

        writer.writerow(data_to_save)

    print(f"结果已保存到: {csv_path}")
    return csv_path

# ==================== 主程序 =====================
def main():
    # ================== Step 0: 加载指定响应文件 ==================
    print("=" * 60)
    print("开始分析单个响应文件")
    print("=" * 60)
    print(f"文件路径: {PICKLE_PATH}")

    if not os.path.exists(PICKLE_PATH):
        print("错误: 指定的pickle文件不存在！")
        exit()

    # ================== Step 1: 加载和处理响应信号 ==================
    print("\n" + "-" * 40)
    print("Step 1: 加载响应信号")
    all_resp_voltage, ref_time_resp = load_single_response_file(PICKLE_PATH)

    if all_resp_voltage is None or len(all_resp_voltage) == 0:
        print("错误: 未加载到有效响应数据")
        exit()

    print(f"成功加载 {all_resp_voltage.shape[0]} 条响应波形")
    print(f"原始每条波形采样点数: {all_resp_voltage.shape[1]} 点")

    # 去偏置
    all_resp_corrected = remove_dc_offset_individual(all_resp_voltage, ref_time_resp)

    # ===== 剔除第一次测量 =====
    if all_resp_corrected.shape[0] > 1:
        all_resp_corrected = all_resp_corrected[1:]
    else:
        print(f"警告: 只有一组数据，无法剔除")

    voltage_mV_avg_resp = np.mean(all_resp_corrected, axis=0)
    voltage_mV_std_resp = np.std(all_resp_corrected, axis=0)

    time_us_resp = ref_time_resp
    time_s_resp = time_us_resp * 1e-6

    if USE_RESPONSE_WINDOW:
        # 响应时间窗口：仅保留指定时间范围内的数据
        resp_mask = (time_us_resp >= RESPONSE_WINDOW_START) & (time_us_resp <= RESPONSE_WINDOW_END)
        time_us_resp_win = time_us_resp[resp_mask]
        time_s_resp_win = time_s_resp[resp_mask]
        voltage_mV_resp_win = voltage_mV_avg_resp[resp_mask]
        all_resp_win = all_resp_corrected[:, resp_mask]  # 所有波形的窗口部分

        Vpp_window_mV = np.max(voltage_mV_resp_win) - np.min(voltage_mV_resp_win)
        print(f"响应窗口: {RESPONSE_WINDOW_START}-{RESPONSE_WINDOW_END} μs, Vpp = {Vpp_window_mV:.2f} mV")
        print(f"加窗后每条波形采样点数: {all_resp_win.shape[1]} 点")
    else:
        # 不使用响应窗口：使用全部响应数据
        time_us_resp_win = time_us_resp
        time_s_resp_win = time_s_resp
        voltage_mV_resp_win = voltage_mV_avg_resp
        all_resp_win = all_resp_corrected

        Vpp_window_mV = np.max(voltage_mV_resp_win) - np.min(voltage_mV_resp_win)
        print(f"使用完整响应信号, Vpp = {Vpp_window_mV:.2f} mV")
        print(f"不加窗每条波形采样点数: {all_resp_win.shape[1]} 点")

    # ================== Step 2: 平均响应 FFT ==================
    print("\n" + "-" * 40)
    print("Step 2: 平均响应频谱分析")

    fft_resp_avg = average_response_fft(
        all_resp_win=all_resp_win,
        time_resp_win=time_s_resp_win,
        max_freq=FFT_MAX_FREQ,
        zero_padding=FFT_ZERO_PADDING
    )

    # ================== Step 3: 带宽分析 ==================
    print("\n" + "-" * 40)
    print("Step 3: 带宽分析")

    bandwidth_resp = calculate_bandwidth(
        fft_resp_avg['freqs_valid'],
        fft_resp_avg['fft_norm'],
        fft_resp_avg['fft_dB'],
        thresholds={'3dB': 0.707},
        fmin = None #限制频率搜索下限
    )

    if '3dB_bw' in bandwidth_resp and bandwidth_resp['3dB_bw'] is not None:
        cf_resp = bandwidth_resp.get('3dB_center_hz')
        if cf_resp and cf_resp > 0:
            bw_resp = bandwidth_resp['3dB_bw']
            bandwidth_resp['3dB_relative_bandwidth_percent'] = (bw_resp / cf_resp) * 100

    # 计算平均响应频谱的峰值
    resp_peak_idx = np.argmax(fft_resp_avg['fft_norm'])
    resp_peak_freq_hz = fft_resp_avg['freqs_valid'][resp_peak_idx]
    resp_peak_freq_mhz = resp_peak_freq_hz / 1e6
    resp_peak_dB = fft_resp_avg['fft_dB'][resp_peak_idx]

    # ================== Step 4: 绘图 ==================
    print("\n" + "-" * 40)
    print("Step 4: 生成图表")

    # 4.1 原始响应波形
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_us_resp, voltage_mV_avg_resp, linewidth=1.2, color='#0072BD')
    ax1.set_xlabel('Time (μs)', fontsize=13.5)
    ax1.set_ylabel('Received Voltage (mV)', fontsize=13.5)
    ax1.set_title(f'Time-Domain PMUT Acoustic Output (Measured by Hydrophone)\nDevice {die_id}', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if USE_RESPONSE_WINDOW:
        ax1.axvline(x=RESPONSE_WINDOW_START, color='red', linestyle='--', alpha=0.8)
        ax1.axvline(x=RESPONSE_WINDOW_END, color='red', linestyle='--', alpha=0.8)

    legend_elements = [
        plt.Line2D([0], [0], color='#0072BD', lw=1.2, label='Average Signal'),
        plt.Line2D([0], [0], color='white', lw=0, label=f'Vpp: {Vpp_window_mV:.2f} mV')
    ]
    if USE_RESPONSE_WINDOW:
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linestyle='--', lw=1.2, alpha=0.8,
                       label=f'Window({RESPONSE_WINDOW_START}-{RESPONSE_WINDOW_END} μs)')
        )
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
               frameon=True, fancybox=True, framealpha=0.8, fontsize=11)
    plt.tight_layout()
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "average_voltage_waveform"), fig=fig1, formats=FIG_FORMATS)

    # 4.1a 时域波形叠加图
    fig_time_overlay, ax_time_overlay = plt.subplots(figsize=(10, 6))

    # 计算每条波形的Vpp（使用指定窗口）
    vpp_values = []
    for i in range(all_resp_corrected.shape[0]):
        if USE_RESPONSE_WINDOW:
            # 如果使用窗口，计算窗口内的Vpp
            resp_mask = (time_us_resp >= RESPONSE_WINDOW_START) & (time_us_resp <= RESPONSE_WINDOW_END)
            signal_window = all_resp_corrected[i, resp_mask]
            vpp = np.max(signal_window) - np.min(signal_window)
        else:
            # 如果不用窗口，计算完整波形的Vpp
            vpp = np.max(all_resp_corrected[i]) - np.min(all_resp_corrected[i])
        vpp_values.append(vpp)

    # 限制显示的波形数量，避免过于密集
    n_traces_to_plot_time = min(20, all_resp_corrected.shape[0])

    # 绘制时域波形叠加
    for i in range(n_traces_to_plot_time):
        ax_time_overlay.plot(time_us_resp, all_resp_corrected[i],
                             linewidth=1.0)

    ax_time_overlay.set_xlabel('Time (μs)', fontsize=13.5)
    ax_time_overlay.set_ylabel('Received Voltage (mV)', fontsize=13.5)
    ax_time_overlay.set_title(f'Time-Domain Waveforms Overlay\nDevice {die_id}', fontsize=16, fontweight='bold')
    ax_time_overlay.grid(True, alpha=0.3)

    # 创建图例，显示每条波形的Vpp值
    legend_elements_time = []

    # 添加前20条波形的Vpp信息到图例
    n_legend_items = min(20, n_traces_to_plot_time)

    for i in range(n_legend_items):
        legend_elements_time.append(
            plt.Line2D([0], [0], color=ax_time_overlay.lines[i].get_color(), lw=1.0,
                       label=f'Trace {i + 1}: Vpp = {vpp_values[i]:.2f} mV')
        )

    # 添加总体统计
    legend_elements_time.append(
        plt.Line2D([0], [0], color='white', lw=0,
                   label=f'Total traces: {all_resp_corrected.shape[0]}')
    )
    legend_elements_time.append(
        plt.Line2D([0], [0], color='white', lw=0,
                   label=f'Avg Vpp: {np.mean(vpp_values):.2f} mV')
    )
    legend_elements_time.append(
        plt.Line2D([0], [0], color='white', lw=0,
                   label=f'Std Vpp: {np.std(vpp_values):.2f} mV')
    )
    legend_elements_time.append(
        plt.Line2D([0], [0], color='white', lw=0,
                   label=f'Window: {RESPONSE_WINDOW_START}-{RESPONSE_WINDOW_END} μs')
    )

    ax_time_overlay.legend(handles=legend_elements_time, loc='upper left', bbox_to_anchor=(1.02, 1),
                           frameon=True, fancybox=True, framealpha=0.8, fontsize=9)
    plt.tight_layout()

    # 保存时域叠加图
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "time_domain_waveforms_overlay"),
                         fig=fig_time_overlay, formats=FIG_FORMATS)

    # 4.2 响应窗口内波形（单独绘制）
    if USE_RESPONSE_WINDOW:
        fig1b_window, ax1b_window = plt.subplots(figsize=(10, 6))
        ax1b_window.plot(time_us_resp_win, voltage_mV_resp_win, linewidth=1.2, color='#0072BD')
        window_vpp = np.max(voltage_mV_resp_win) - np.min(voltage_mV_resp_win)
        ax1b_window.set_xlabel('Time (μs)', fontsize=13.5)
        ax1b_window.set_ylabel('Received Voltage (mV)', fontsize=13.5)
        ax1b_window.set_title(f'Time-Domain PMUT Acoustic Output (Measured by Hydrophone)\nDevice {die_id}',
                              fontsize=16, fontweight='bold')
        ax1b_window.grid(True, alpha=0.3)
        legend_elements_window = [
            plt.Line2D([0], [0], color='#0072BD', lw=1.2, label='Average Signal'),
            plt.Line2D([0], [0], color='white', lw=0, label=f'Vpp: {window_vpp:.2f} mV')
        ]
        ax1b_window.legend(handles=legend_elements_window, loc='upper left', bbox_to_anchor=(1.02, 1),
                           frameon=True, fancybox=True, framealpha=0.8, fontsize=11)
        plt.tight_layout()
        save_fig_all_formats(os.path.join(OUTPUT_DIR, "response_window_waveform"), fig=fig1b_window, formats=FIG_FORMATS)

    # 4.3 单次响应频谱叠加图
    fig_response_fft, ax_response_fft = plt.subplots(figsize=(10, 6))

    # 绘制单次响应频谱叠加 - 使用科学的色彩叠加
    n_traces_to_plot = min(20, all_resp_win.shape[0])  # 限制显示的数量，避免过于密集

    # 创建色彩映射
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_traces_to_plot))

    # 存储每个频谱的峰值信息
    peak_freqs = []
    peak_dBs = []

    for i in range(n_traces_to_plot):
        fft_single = perform_fft_analysis(
            time_s_resp_win,
            all_resp_win[i],
            max_freq=FFT_MAX_FREQ,
            zero_padding_factor=FFT_ZERO_PADDING
        )
        mag = np.abs(fft_single['fft_complex'])
        mag_norm = mag / (np.max(mag) + 1e-15)
        mag_dB = 20 * np.log10(mag_norm + 1e-15)

        # 找到峰值
        peak_idx = np.argmax(mag_norm)
        peak_freq_hz = fft_single['freqs_valid'][peak_idx]
        peak_freq_mhz = peak_freq_hz / 1e6
        peak_freqs.append(peak_freq_mhz)
        peak_dBs.append(mag_dB[peak_idx])

        # 绘制频谱
        ax_response_fft.plot(fft_single['freqs_valid'] / 1e6, mag_dB,
                             color=colors[i], alpha=0.7, linewidth=1.0, zorder=1)

    # 创建图例元素
    legend_elements_resp = []

    if n_traces_to_plot > 1:
        # 添加峰值统计信息（可选）
        pass

    # 添加单个峰值统计（最多前20个）
    for i in range(min(20, n_traces_to_plot)):
        legend_elements_resp.append(
            plt.Line2D([0], [0], color=colors[i], lw=1.0, alpha=0.7,
                       label=f'Trace {i + 1} Peak: {peak_freqs[i]:.2f} MHz')
        )

    ax_response_fft.set_xlabel('Frequency (MHz)', fontsize=13.5)
    ax_response_fft.set_ylabel('Normalized Magnitude (dB)', fontsize=13.5)
    ax_response_fft.set_title(
        f'Frequency Spectrum of Received Signal (n={n_traces_to_plot} traces)\nDevice {die_id}',
        fontsize=16, fontweight='bold')
    ax_response_fft.set_xlim(0, FFT_MAX_FREQ/1e6)
    ax_response_fft.set_ylim(-70, 5)
    ax_response_fft.grid(True, alpha=0.3, zorder=0)

    # 设置图例
    ax_response_fft.legend(handles=legend_elements_resp, loc='upper left', bbox_to_anchor=(1.02, 1),
                           frameon=True, fancybox=True, framealpha=0.8, fontsize=8)

    plt.tight_layout()
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "response_fft_analysis"), fig=fig_response_fft,
                         formats=FIG_FORMATS)

    # 4.4 平均频谱图（单独，无叠加）
    fig_avg_spectrum, ax_avg_spectrum = plt.subplots(figsize=(10, 6))
    ax_avg_spectrum.plot(fft_resp_avg['freqs_valid_mhz'], fft_resp_avg['fft_dB'],
                          color='#0072BD', linewidth=2.0, label='Average Response Spectrum')

    ax_avg_spectrum.axhline(-3, color='orange', linestyle='--', linewidth=1.2, alpha=0.7)

    ax_avg_spectrum.scatter(resp_peak_freq_mhz, resp_peak_dB, color='red', s=40, edgecolors='black', zorder=10)

    legend_elements_avg = [
        plt.Line2D([0], [0], color='#0072BD', lw=1.5, label='Average Response Spectrum'),
        plt.Line2D([0], [0], color='orange', linestyle='--', lw=1.2, label='-3dB Threshold'),
        plt.Line2D([0], [0], marker='o', color='red', markersize=5,
                   label=f'Peak: {resp_peak_freq_mhz:.2f} MHz ({resp_peak_dB:.1f} dB)')
    ]

    for key in ['3dB_low_hz', '3dB_high_hz']:
        if key in bandwidth_resp and bandwidth_resp[key] is not None:
            f_mhz = bandwidth_resp[key] / 1e6
            ax_avg_spectrum.axvline(f_mhz, color='purple', linestyle=':', linewidth=1.5)
            legend_elements_avg.append(plt.Line2D([0], [0], color='purple', linestyle=':', lw=1.5,
                                                   label=f'{key.replace("_hz", "")}: {f_mhz:.2f} MHz'))

    if '3dB_center_hz' in bandwidth_resp and bandwidth_resp['3dB_center_hz'] is not None:
        f_center = bandwidth_resp['3dB_center_hz'] / 1e6
        dB_center = np.interp(f_center, fft_resp_avg['freqs_valid_mhz'], fft_resp_avg['fft_dB'])
        ax_avg_spectrum.scatter(f_center, dB_center, color='orange', s=40, edgecolors='black', zorder=10)
        legend_elements_avg.append(plt.Line2D([0], [0], marker='o', color='orange', markersize=5,
                                               label=f'-3dB Center: {f_center:.2f} MHz'))

    if '3dB_bw' in bandwidth_resp and bandwidth_resp['3dB_bw'] is not None:
        legend_elements_avg.append(plt.Line2D([0], [0], color='white', lw=0,
                                               label=f'-3dB BW: {bandwidth_resp["3dB_bw"] / 1e6:.2f} MHz'))

    if bandwidth_resp.get('3dB_relative_bandwidth_percent') is not None:
        rel_bw_resp = bandwidth_resp["3dB_relative_bandwidth_percent"]
        legend_elements_avg.append(plt.Line2D([0], [0], color='white', lw=0,
                                               label=f'Relative BW: {rel_bw_resp:.1f}%'))

    ax_avg_spectrum.set_xlabel('Frequency (MHz)', fontsize=13.5)
    ax_avg_spectrum.set_ylabel('Normalized Magnitude (dB)', fontsize=13.5)
    ax_avg_spectrum.set_title(f'Normalized Frequency Spectrum (Measured by Hydrophone)\nDevice {die_id}', fontsize=16, fontweight='bold')
    ax_avg_spectrum.set_xlim(0, FFT_MAX_FREQ/1e6)
    ax_avg_spectrum.set_ylim(-70, 5)
    ax_avg_spectrum.grid(True, alpha=0.3)
    ax_avg_spectrum.legend(handles=legend_elements_avg, loc='upper left', bbox_to_anchor=(1.02, 1),
                           frameon=True, fancybox=True, framealpha=0.8, fontsize=8)
    plt.tight_layout()
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "average_response_spectrum"), fig=fig_avg_spectrum, formats=FIG_FORMATS)

    # ================== Step 5: 结果摘要 ==================
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)
    print(f"响应波形数量: {all_resp_win.shape[0]}")
    print(f"响应窗口: {RESPONSE_WINDOW_START} ～ {RESPONSE_WINDOW_END} μs")
    print(f"窗口Vpp: {Vpp_window_mV:.2f} mV")

    print(f"\n【响应频谱】")
    print(f"  峰值频率: {resp_peak_freq_mhz:.2f} MHz")
    if '3dB_bw' in bandwidth_resp and bandwidth_resp['3dB_bw'] is not None:
        print(f"  -3dB 带宽: {bandwidth_resp['3dB_bw'] / 1e6:.2f} MHz")
        print(f"  中心频率: {bandwidth_resp['3dB_center_hz'] / 1e6:.2f} MHz")
        if '3dB_relative_bandwidth_percent' in bandwidth_resp:
            print(f"  相对带宽: {bandwidth_resp['3dB_relative_bandwidth_percent']:.2f}%")
    else:
        print("  -3dB 带宽: 未检测到有效峰值")

    # ================== Step 6: 保存结果到CSV ==================
    print("\n" + "-" * 40)
    print("Step 6: 保存结果到CSV文件")

    results = {
        'response_files_count': all_resp_win.shape[0],
        'response_window_vpp': Vpp_window_mV,
    }

    response_spectrum = {
        'peak_freq_MHz': resp_peak_freq_mhz,
        'peak_mag_dB': resp_peak_dB,
    }
    if '3dB_bw' in bandwidth_resp and bandwidth_resp['3dB_bw'] is not None:
        response_spectrum['3dB_bw_MHz'] = bandwidth_resp['3dB_bw'] / 1e6
        response_spectrum['3dB_center_freq_MHz'] = bandwidth_resp.get('3dB_center_hz', 0) / 1e6
        response_spectrum['relative_bw_percent'] = bandwidth_resp.get('3dB_relative_bandwidth_percent', 0)
    results['response_spectrum'] = response_spectrum

    csv_file = save_results_to_csv(results, OUTPUT_DIR, f"response_analysis_results_{die_id}.csv")

    print("\n图表已保存到:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
