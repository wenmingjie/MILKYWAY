"""
CSV 文件夹批量分析器
====================
功能：
1. 从两个文件夹分别加载所有 CSV 文件（频域/时域）
2. 对频域文件夹所有信号做 FFT 复数平均
3. 对时域文件夹所有信号取时域平均
4. 计算时域参数（Vpp, 包络, 脉冲宽度）
5. 计算频域参数（Fc, BW, HD2）
6. 绘制双轴叠加图（含信息框）
7. 保存 PNG 图片
"""
import os
import numpy as np

# 公共模块
from io.csv_io import load_csv_signals, load_csv_average
from analysis.waveform import remove_dc_offset, apply_time_window, compute_envelope, pulse_duration
from analysis.spectrum import average_spectrum_complex, calculate_bandwidth
from plot.figure import plot_time_freq_dual_axis_with_boxes
from plot.style import set_style


class CSVBatchAnalyzer:
    """CSV 文件夹批量分析器"""

    def __init__(self, config):
        """
        参数:
            config: dict，包含：
                - folder_1cycle: 频域 CSV 文件夹路径
                - folder_3cycle: 时域 CSV 文件夹路径
                - die_id: 器件 ID（用于标题）
                - window_1cycle: (start, end) μs
                - window_3cycle: (start, end) μs
                - fft_max_freq: FFT 最大频率 (Hz)
                - fft_zero_padding: 零填充倍数
                - output_dir: 输出目录
                - title_prefix: 标题前缀（可选）
        """
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """执行批量分析"""
        set_style()
        print("="*60)
        print("CSV 文件夹批量分析（频域复数平均 + 时域平均）")
        print("="*60)

        # 1. 加载数据
        signals1, time1 = load_csv_signals(self.config['folder_1cycle'])
        sig3_avg, time3 = load_csv_average(self.config['folder_3cycle'])
        if signals1 is None or sig3_avg is None:
            print("数据加载失败")
            return

        print(f"频域信号数量: {signals1.shape[0]}")
        print(f"时域信号数量: {sig3_avg.shape[0]}")

        # 2. 去直流（时域）
        sig3_avg = remove_dc_offset(sig3_avg, time3)

        # 3. 应用窗口
        # 频域：所有信号分别截窗
        signals1_win = []
        time1_win = None
        for sig in signals1:
            sig_win, t_win = apply_time_window(sig, time1, self.config['window_1cycle'])
            signals1_win.append(sig_win)
            if time1_win is None:
                time1_win = t_win
        signals1_win = np.array(signals1_win)

        # 时域：平均信号截窗
        sig3_win, time3_win = apply_time_window(sig3_avg, time3, self.config['window_3cycle'])

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
        fft_avg = average_spectrum_complex(
            signals1_win, time1_win,
            self.config['fft_max_freq'],
            self.config['fft_zero_padding']
        )

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

        # 6. HD2 计算
        hd2_dBc = None
        hd2_freq = 2 * bw_results['peak_freq_hz']
        if hd2_freq <= self.config['fft_max_freq']:
            hd2_dB = np.interp(hd2_freq, fft_avg['freqs_valid'], fft_avg['fft_dB'])
            hd2_dBc = hd2_dB - peak_dB

        # 构建频域信息
        low = bw_results.get('3dB_low_hz')
        high = bw_results.get('3dB_high_hz')
        freq_info = (f"Peak Freq. : {peak_freq_mhz:.2f} MHz\n"
                     f"Center Freq. : {center_hz/1e6:.2f} MHz\n" if center_hz else ""
                     f"-3dB Low : {low/1e6:.2f} MHz\n" if low else ""
                     f"-3dB High : {high/1e6:.2f} MHz\n" if high else ""
                     f"-3dB BW : {bw_hz/1e6:.2f} MHz\n" if bw_hz else ""
                     f"Rel. BW : {rel_bw:.1f} %\n"
                     f"HD2 : {hd2_dBc:.1f} dBc" if hd2_dBc is not None else "HD2 : N/A")

        # 7. 绘图
        title_prefix = self.config.get('title_prefix', 'Transmit Response Waveform')
        title = f"{title_prefix} of Device {self.config['die_id']}"
        save_path = os.path.join(self.output_dir,
                                 f"{self.config['die_id']}_avg_spectrum.png")

        fig, _, _ = plot_time_freq_dual_axis_with_boxes(
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
            freq_xlim=(0, self.config['fft_max_freq']/1e6),
            freq_ylim=(-60, 5),
            save_path=save_path
        )

        # 8. 打印摘要
        print("\n【时域参数】")
        print(f"  Vpp = {vpp:.2f} mV")
        print(f"  Peak = {peak_voltage:.2f} mV")
        print(f"  -6dB Width = {dur6_str} μs")
        print(f"  -20dB Width = {dur20_str} μs")

        print("\n【频域参数】")
        print(f"  信号数量 = {signals1_win.shape[0]}")
        print(f"  Peak freq. = {peak_freq_mhz:.3f} MHz")
        if center_hz:
            print(f"  Center freq. = {center_hz/1e6:.3f} MHz")
        if bw_hz:
            print(f"  -3dB BW = {bw_hz/1e6:.3f} MHz")
        print(f"  Rel. BW = {rel_bw:.2f}%")
        if hd2_dBc is not None:
            print(f"  HD2 = {hd2_dBc:.1f} dBc")

        print(f"\n图片已保存: {save_path}")
        print("="*60)

        return {
            'vpp': vpp,
            'peak_voltage': peak_voltage,
            'peak_freq_mhz': peak_freq_mhz,
            'center_freq_mhz': center_hz/1e6 if center_hz else np.nan,
            'bw_mhz': bw_hz/1e6 if bw_hz else np.nan,
            'rel_bw_percent': rel_bw,
            'hd2_dBc': hd2_dBc,
        }
