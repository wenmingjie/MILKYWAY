import matplotlib.pyplot as plt
import numpy as np
from .style import set_style, COLOR_TIME, COLOR_FREQ, COLOR_PEAK, COLOR_THRESHOLD, COLOR_BAND

def plot_time_freq_dual_axis_with_boxes(
    time_us, signal,
    freqs_mhz, spectrum_dB,
    peak_freq_mhz, peak_dB,
    bandwidth_results,
    hd2_dBc=None,
    title=None,
    time_label='Time (μs)', freq_label='Frequency (MHz)',
    time_color='#0072BD', freq_color='#D95319',
    figsize=(10,6),
    freq_xlim=(0, 30), freq_ylim=(-60, 5),
    time_ylim=None,
    time_info_text=None,
    freq_info_text=None,
    bbox_style=None,
    save_path=None
):
    """
    绘制双轴叠加图（左下时域，右上频域），并带两个信息框（置于时域轴右侧）。
    完全匹配原脚本绘图风格。

    参数:
        time_us : ndarray
        signal : ndarray 时域信号 (mV)
        freqs_mhz : ndarray 频率轴 (MHz)
        spectrum_dB : ndarray 频谱幅度 (dB)
        peak_freq_mhz, peak_dB : 峰值点
        bandwidth_results : dict 包含 '3dB_low_hz', '3dB_high_hz', '3dB_center_hz', '3dB_bw'
        hd2_dBc : float or None, 二次谐波相对幅度 (dBc)
        title : str, 标题
        time_label, freq_label : 轴标签
        time_color, freq_color : 主颜色
        freq_xlim, freq_ylim : 频域轴范围
        time_ylim : 时域轴范围 (若None自动)
        time_info_text, freq_info_text : 自定义信息框文本（若提供则覆盖自动生成）
        bbox_style : 信息框样式字典
        save_path : 保存路径（若提供则保存，否则只返回fig）
    返回:
        fig, ax_time, ax_freq
    """
    set_style()
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # --- 时域轴（左下） ---
    ax_time = fig.add_axes([0.10, 0.12, 0.68, 0.76])
    ax_time.set_xlim(time_us[0], time_us[-1])
    ax_time.set_xlabel(time_label, fontsize=13.5, color=time_color)
    ax_time.set_ylabel("Voltage (mV)", fontsize=13.5, color=time_color)
    ax_time.tick_params(axis='both', direction='in', length=5, width=1.2, labelsize=11)
    ax_time.tick_params(axis='y', colors=time_color)
    ax_time.tick_params(axis='x', colors=time_color)
    for s in ax_time.spines.values():
        s.set_linewidth(1.2)
    ax_time.spines['left'].set_color(time_color)
    ax_time.spines['bottom'].set_color(time_color)
    ax_time.spines['top'].set_color(freq_color)
    ax_time.spines['right'].set_color(freq_color)
    ax_time.grid(True, linestyle='--', linewidth=0.5, alpha=0.25)
    if time_ylim is not None:
        ax_time.set_ylim(time_ylim)

    # --- 频域轴（右上） ---
    ax_freq = fig.add_axes(ax_time.get_position(), frameon=False)
    ax_freq.set_xlim(freq_xlim[0], freq_xlim[1])
    ax_freq.set_ylim(freq_ylim[0], freq_ylim[1])
    ax_freq.xaxis.tick_top()
    ax_freq.xaxis.set_label_position('top')
    ax_freq.yaxis.tick_right()
    ax_freq.yaxis.set_label_position('right')
    ax_freq.set_xlabel(freq_label, fontsize=13.5, color=freq_color)
    ax_freq.set_ylabel("Normalized Magnitude (dB)", fontsize=13.5, color=freq_color)
    ax_freq.tick_params(axis='x', direction='in', length=5, width=1.2, labelsize=11,
                        color=freq_color, labelcolor=freq_color)
    ax_freq.tick_params(axis='y', direction='in', length=5, width=1.2, labelsize=11,
                        color=freq_color, labelcolor=freq_color)
    ax_freq.spines['top'].set_visible(True)
    ax_freq.spines['right'].set_visible(True)
    ax_freq.spines['top'].set_color(freq_color)
    ax_freq.spines['right'].set_color(freq_color)
    ax_freq.spines['top'].set_linewidth(1.2)
    ax_freq.spines['right'].set_linewidth(1.2)
    ax_freq.spines['left'].set_visible(False)
    ax_freq.spines['bottom'].set_visible(False)

    ax_freq.set_zorder(1)
    ax_time.set_zorder(2)
    ax_time.patch.set_alpha(0)

    # ---- 绘制频域 ----
    ax_freq.plot(freqs_mhz, spectrum_dB, color=freq_color, lw=2.2, zorder=1)
    ax_freq.scatter(peak_freq_mhz, peak_dB, color='red', s=45, zorder=3)
    # 中心频率点
    center_hz = bandwidth_results.get('3dB_center_hz')
    if center_hz is not None:
        center_mhz = center_hz / 1e6
        center_dB = np.interp(center_mhz, freqs_mhz, spectrum_dB)
        ax_freq.scatter(center_mhz, center_dB, color='orange', s=45, zorder=3)
    # -3dB 线
    peak_val = np.max(spectrum_dB)
    ax_freq.axhline(peak_val - 3, color=freq_color, ls=':', lw=1.2)
    # 高低频垂直线
    low = bandwidth_results.get('3dB_low_hz')
    high = bandwidth_results.get('3dB_high_hz')
    if low is not None:
        ax_freq.axvline(low/1e6, color=freq_color, ls='--', lw=1.0)
    if high is not None:
        ax_freq.axvline(high/1e6, color=freq_color, ls='--', lw=1.0)

    # ---- 绘制时域 ----
    ax_time.plot(time_us, signal, color=time_color, lw=2.5, zorder=100)
    # 峰值点
    peak_idx = np.argmax(np.abs(signal))
    peak_time = time_us[peak_idx]
    peak_val_signal = signal[peak_idx]
    ax_time.scatter(peak_time, peak_val_signal, color='red', s=45, zorder=101)

    # ---- 信息框 ----
    if bbox_style is None:
        bbox_style = dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="black", linewidth=1.0, alpha=0.5)
    # 时域信息
    if time_info_text is None:
        # 自动生成（需要外部提供 Vpp, 峰值, dur6, dur20 等，这里简单占位）
        # 为保持灵活性，由调用者传入
        time_info_text = "Vpp : ...\nPeak : ...\n-6dB Width : ...\n-20dB Width : ..."
    ax_time.text(0.985, 0.98, time_info_text, transform=ax_time.transAxes,
                 ha='right', va='top', fontsize=10, family='Arial', bbox=bbox_style, zorder=200)

    # 频域信息
    if freq_info_text is None:
        bw = bandwidth_results.get('3dB_bw', 0)
        rel_bw = (bw / center_hz * 100) if center_hz and center_hz > 0 else 0
        freq_info_text = (f"Peak Freq. : {peak_freq_mhz:.2f} MHz\n"
                          f"Center Freq. : {center_hz/1e6:.2f} MHz\n"
                          f"-3dB Low : {low/1e6:.2f} MHz\n" if low is not None else ""
                          f"-3dB High : {high/1e6:.2f} MHz\n" if high is not None else ""
                          f"-3dB BW : {bw/1e6:.2f} MHz\n"
                          f"Rel. BW : {rel_bw:.1f} %\n"
                          f"HD2 : {hd2_dBc:.1f} dBc" if hd2_dBc is not None else "")
    ax_time.text(0.985, 0.78, freq_info_text, transform=ax_time.transAxes,
                 ha='right', va='top', fontsize=10, family='Arial', bbox=bbox_style, zorder=200)

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig, ax_time, ax_freq
