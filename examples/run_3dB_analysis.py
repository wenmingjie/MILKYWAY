"""
piclke 分析主程序
功能：加载 Pickle，时域窗口截取，FFT 复数平均，带宽分析，绘图保存，CSV 导出
新增：时域频域双轴叠加图（含信息框）
"""
import os
import numpy as np
import pandas as pd
from io.pickle_io import load_pickle_waveforms
from io.csv_io import append_results_to_csv
from analysis.waveform import remove_dc_offset, apply_time_window, compute_vpp, compute_mean_std, compute_envelope, pulse_duration
from analysis.spectrum import average_spectrum_complex, calculate_bandwidth
from plot.waveform import plot_wave_with_window, plot_wave_overlay
from plot.spectrum import plot_avg_spectrum, plot_spectrum_overlay
from plot.figure import plot_time_freq_dual_axis_with_boxes
from plot.save import save_fig_all_formats
from plot.style import set_style

# ==================== 配置参数 =====================
BASE_DIR = r'Y:\Measurements\2026\20260722'
FILE_NAME = r'MKW_STB3SP11W00_R3C0_L_f10.0MHz_Txallchannelpulse_HV20Vpp_1cycles_d5mm_11times_gaus_T24.8C_zshu_20260723.pickle'
DIE_ID = 'STB3SP11W00_R3C0_L'
PICKLE_PATH = os.path.join(BASE_DIR, FILE_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, f'analysis_{DIE_ID}_guas-5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_FORMATS = ['png']          # 可增加 'pdf', 'emf'
USE_WINDOW = True
WINDOW_START_US = 3.3
WINDOW_END_US = 4.3
FFT_MAX_FREQ = 20e6
FFT_ZERO_PADDING = 4

# ==================== 主流程 =====================
def main():
    set_style()
    print("="*60)
    print("Echo 分析 (重构版) - 含双轴叠加图")
    print("="*60)

    # 1. 加载数据
    signals_mV, time_us, n = load_pickle_waveforms(PICKLE_PATH, channel='ch2')
    if signals_mV is None:
        return

    # 2. 去除直流偏置
    signals_corr = remove_dc_offset(signals_mV, time_us, baseline_region=(0,1))

    # 3. 剔除第一次测量（若有多条）
    if signals_corr.shape[0] > 1:
        signals_corr = signals_corr[1:]
    else:
        print("警告: 仅一条波形，未剔除")

    # 4. 计算平均信号
    signal_avg, signal_std = compute_mean_std(signals_corr, axis=0)

    # 5. 应用时间窗口（如有）
    if USE_WINDOW:
        window_us = (WINDOW_START_US, WINDOW_END_US)
        all_win, time_win = apply_time_window(signals_corr, time_us, window_us)
        avg_win, _ = apply_time_window(signal_avg[np.newaxis,:], time_us, window_us)
        avg_win = avg_win[0]
    else:
        all_win, time_win = signals_corr, time_us
        avg_win = signal_avg

    vpp_window = compute_vpp(avg_win)
    print(f"窗口 Vpp = {vpp_window:.2f} mV")

    # 6. FFT 复数平均
    fft_avg = average_spectrum_complex(all_win, time_win, FFT_MAX_FREQ, FFT_ZERO_PADDING)

    # 7. 带宽分析
    bw_results = calculate_bandwidth(
        fft_avg['freqs_valid'],
        fft_avg['fft_norm'],
        fft_avg['fft_dB'],
        thresholds={'3dB': 1/np.sqrt(2)},
        fmin=None
    )
    # 添加相对带宽
    center = bw_results.get('3dB_center_hz')
    bw = bw_results.get('3dB_bw')
    if center and center>0 and bw is not None:
        bw_results['3dB_relative_bandwidth_percent'] = (bw / center) * 100

    peak_freq_mhz = bw_results['peak_freq_mhz']
    peak_dB = bw_results['peak_amplitude_dB']

    # 8. 时域包络参数 (基于窗口内平均信号)
    envelope = compute_envelope(avg_win)
    dur_6dB = pulse_duration(time_win, envelope, -6)
    dur_20dB = pulse_duration(time_win, envelope, -20)
    dur6_str = "N/A" if dur_6dB is None else f"{dur_6dB:.2f}"
    dur20_str = "N/A" if dur_20dB is None else f"{dur_20dB:.2f}"
    peak_voltage = np.max(np.abs(avg_win))
    time_info = (f"Vpp : {vpp_window:.2f} mV\n"
                 f"Peak : {peak_voltage:.2f} mV\n"
                 f"-6 dB Width : {dur6_str} μs\n"
                 f"-20 dB Width : {dur20_str} μs")

    # 9. HD2 计算
    hd2_dBc = None
    hd2_freq = 2 * bw_results['peak_freq_hz']
    if hd2_freq <= FFT_MAX_FREQ:
        hd2_dB = np.interp(hd2_freq, fft_avg['freqs_valid'], fft_avg['fft_dB'])
        hd2_dBc = hd2_dB - peak_dB

    # 10. 频域信息文本
    low = bw_results.get('3dB_low_hz')
    high = bw_results.get('3dB_high_hz')
    bw_hz = bw_results.get('3dB_bw')
    rel_bw = bw_results.get('3dB_relative_bandwidth_percent', 0)
    freq_info = (f"Peak Freq. : {peak_freq_mhz:.2f} MHz\n"
                 f"Center Freq. : {center/1e6:.2f} MHz\n" if center else ""
                 f"-3dB Low : {low/1e6:.2f} MHz\n" if low else ""
                 f"-3dB High : {high/1e6:.2f} MHz\n" if high else ""
                 f"-3dB BW : {bw_hz/1e6:.2f} MHz\n" if bw_hz else ""
                 f"Rel. BW : {rel_bw:.1f} %\n"
                 f"HD2 : {hd2_dBc:.1f} dBc" if hd2_dBc is not None else "HD2 : N/A")

    # ================== 绘图（原有独立图） ==================
    # (a) 平均波形 + 窗口
    fig1 = plot_wave_with_window(time_us, signal_avg,
                                 window_start=WINDOW_START_US if USE_WINDOW else None,
                                 window_end=WINDOW_END_US if USE_WINDOW else None,
                                 title=f'Time-Domain PMUT Acoustic Output\nDevice {DIE_ID}',
                                 vpp=vpp_window)
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "average_voltage_waveform"), fig1, FIG_FORMATS)

    # (b) 波形叠加
    vpp_list = [compute_vpp(apply_time_window(signals_corr[i][np.newaxis,:], time_us, (WINDOW_START_US,WINDOW_END_US))[0][0])
                for i in range(signals_corr.shape[0])]
    fig2 = plot_wave_overlay(time_us, signals_corr, vpp_list, max_traces=20,
                             window_us=(WINDOW_START_US,WINDOW_END_US) if USE_WINDOW else None,
                             title=f'Time-Domain Waveforms Overlay\nDevice {DIE_ID}')
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "time_domain_waveforms_overlay"), fig2, FIG_FORMATS)

    # (c) 窗口内平均波形（若使用了窗口）
    if USE_WINDOW:
        fig3 = plot_wave_with_window(time_win, avg_win,
                                     title=f'Time-Domain PMUT Acoustic Output (windowed)\nDevice {DIE_ID}',
                                     vpp=vpp_window)
        save_fig_all_formats(os.path.join(OUTPUT_DIR, "response_window_waveform"), fig3, FIG_FORMATS)

    # (d) 单次频谱叠加 (最多20条)
    spec_list = []
    for i in range(min(20, all_win.shape[0])):
        from analysis.spectrum import perform_fft_analysis
        res = perform_fft_analysis(time_win*1e-6, all_win[i], FFT_MAX_FREQ, FFT_ZERO_PADDING)
        spec_list.append(res['fft_dB'])
    fig4 = plot_spectrum_overlay(fft_avg['freqs_valid_mhz'], spec_list,
                                 title=f'Frequency Spectrum (n={len(spec_list)} traces)\nDevice {DIE_ID}',
                                 xlim=(0, FFT_MAX_FREQ/1e6))
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "response_fft_analysis"), fig4, FIG_FORMATS)

    # (e) 平均频谱
    fig5 = plot_avg_spectrum(fft_avg['freqs_valid_mhz'], fft_avg['fft_dB'],
                             peak_freq_mhz, peak_dB, bw_results,
                             title=f'Normalized Frequency Spectrum\nDevice {DIE_ID}',
                             xlim=(0, FFT_MAX_FREQ/1e6))
    save_fig_all_formats(os.path.join(OUTPUT_DIR, "average_response_spectrum"), fig5, FIG_FORMATS)

    # ================== 新增：时域频域双轴叠加图（含信息框） ==================
    title_dual = f"Transmit Response Waveform of Device {DIE_ID}"
    save_path_dual = os.path.join(OUTPUT_DIR, "time_freq_dual_axis.png")
    fig6, ax_time, ax_freq = plot_time_freq_dual_axis_with_boxes(
        time_us=time_win,                    # 窗口内时间轴
        signal=avg_win,                      # 窗口内平均信号
        freqs_mhz=fft_avg['freqs_valid_mhz'],
        spectrum_dB=fft_avg['fft_dB'],
        peak_freq_mhz=peak_freq_mhz,
        peak_dB=peak_dB,
        bandwidth_results=bw_results,
        hd2_dBc=hd2_dBc,
        title=title_dual,
        time_info_text=time_info,
        freq_info_text=freq_info,
        freq_xlim=(0, FFT_MAX_FREQ/1e6),
        freq_ylim=(-60, 5),
        save_path=save_path_dual
    )
    # 如果需要显示，可以 plt.show() 但为了批处理，可能注释掉
    # plt.show()

    # 11. 保存 CSV
    results = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'die_id': DIE_ID,
        'response_files_count': all_win.shape[0],
        'window_start_us': WINDOW_START_US if USE_WINDOW else 0,
        'window_end_us': WINDOW_END_US if USE_WINDOW else 0,
        'response_window_vpp_mV': vpp_window,
        'fft_max_freq_MHz': FFT_MAX_FREQ/1e6,
        'response_peak_freq_MHz': peak_freq_mhz,
        'response_peak_mag_dB': peak_dB,
        'response_3dB_bw_MHz': bw_results.get('3dB_bw', 0)/1e6 if bw_results.get('3dB_bw') else 0,
        'response_3dB_center_freq_MHz': bw_results.get('3dB_center_hz', 0)/1e6 if bw_results.get('3dB_center_hz') else 0,
        'response_relative_bw_percent': bw_results.get('3dB_relative_bandwidth_percent', 0),
        'hd2_dBc': hd2_dBc if hd2_dBc is not None else np.nan,
        'pulse_duration_6dB_us': dur_6dB if dur_6dB is not None else np.nan,
        'pulse_duration_20dB_us': dur_20dB if dur_20dB is not None else np.nan,
    }
    append_results_to_csv(OUTPUT_DIR, f"response_analysis_results_{DIE_ID}.csv", results)

    print(f"\n图表已保存到: {OUTPUT_DIR}")
    print("新增双轴叠加图: time_freq_dual_axis.png")
    print("="*60)

if __name__ == "__main__":
    main()
