import matplotlib.pyplot as plt
import numpy as np
from .style import set_style, COLOR_FREQ, COLOR_THRESHOLD, COLOR_PEAK, COLOR_BAND

def plot_spectrum_overlay(freqs_mhz, spectrum_dB_list, max_traces=20,
                          title=None, xlabel='Frequency (MHz)', ylabel='Normalized Magnitude (dB)',
                          figsize=(10,6), xlim=(0,20), ylim=(-70,5)):
    """
    绘制多条频谱叠加（单次 FFT 结果）。
    spectrum_dB_list: list of dB arrays, each same length as freqs_mhz
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    n = min(max_traces, len(spectrum_dB_list))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n))
    for i in range(n):
        ax.plot(freqs_mhz, spectrum_dB_list[i], color=colors[i], alpha=0.7, lw=1.0)
    ax.set_xlabel(xlabel, fontsize=13.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_title(title if title else 'Frequency Spectrum Overlay', fontsize=16, fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    # 图例简化，可显示峰值频率等（由调用者决定）
    plt.tight_layout()
    return fig

def plot_avg_spectrum(freqs_mhz, spectrum_dB, peak_freq_mhz, peak_dB, bandwidth_results,
                      title=None, xlabel='Frequency (MHz)', ylabel='Normalized Magnitude (dB)',
                      figsize=(10,6), xlim=(0,20), ylim=(-70,5)):
    """
    绘制平均频谱，标注 -3dB 带宽、峰值、中心频率等。
    bandwidth_results: 来自 calculate_bandwidth 的 dict
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs_mhz, spectrum_dB, color=COLOR_FREQ, lw=2.0, label='Average Spectrum')
    # -3dB 线
    peak_val = np.max(spectrum_dB)
    ax.axhline(peak_val - 3, color=COLOR_THRESHOLD, linestyle='--', lw=1.2, alpha=0.7)
    # 峰值点
    ax.scatter(peak_freq_mhz, peak_dB, color=COLOR_PEAK, s=40, edgecolors='black', zorder=10)
    legend_handles = [
        plt.Line2D([0],[0], color=COLOR_FREQ, lw=1.5, label='Average Spectrum'),
        plt.Line2D([0],[0], color=COLOR_THRESHOLD, linestyle='--', lw=1.2, label='-3dB Threshold'),
        plt.Line2D([0],[0], marker='o', color=COLOR_PEAK, markersize=5, label=f'Peak: {peak_freq_mhz:.2f} MHz ({peak_dB:.1f} dB)')
    ]
    # 带宽边界
    low = bandwidth_results.get('3dB_low_hz')
    high = bandwidth_results.get('3dB_high_hz')
    center = bandwidth_results.get('3dB_center_hz')
    bw = bandwidth_results.get('3dB_bw')
    if low is not None:
        ax.axvline(low/1e6, color=COLOR_BAND, linestyle=':', lw=1.5)
        legend_handles.append(plt.Line2D([0],[0], color=COLOR_BAND, linestyle=':', lw=1.5,
                                         label=f'Low: {low/1e6:.2f} MHz'))
    if high is not None:
        ax.axvline(high/1e6, color=COLOR_BAND, linestyle=':', lw=1.5)
        legend_handles.append(plt.Line2D([0],[0], color=COLOR_BAND, linestyle=':', lw=1.5,
                                         label=f'High: {high/1e6:.2f} MHz'))
    if center is not None:
        center_mhz = center/1e6
        center_dB = np.interp(center_mhz, freqs_mhz, spectrum_dB)
        ax.scatter(center_mhz, center_dB, color=COLOR_THRESHOLD, s=40, edgecolors='black', zorder=10)
        legend_handles.append(plt.Line2D([0],[0], marker='o', color=COLOR_THRESHOLD, markersize=5,
                                         label=f'-3dB Center: {center_mhz:.2f} MHz'))
    if bw is not None:
        legend_handles.append(plt.Line2D([0],[0], color='white', lw=0,
                                         label=f'-3dB BW: {bw/1e6:.2f} MHz'))
    rel_bw = bandwidth_results.get('3dB_relative_bandwidth_percent')
    if rel_bw is not None:
        legend_handles.append(plt.Line2D([0],[0], color='white', lw=0,
                                         label=f'Rel. BW: {rel_bw:.1f}%'))
    ax.set_xlabel(xlabel, fontsize=13.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_title(title if title else 'Normalized Frequency Spectrum', fontsize=16, fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02,1),
              frameon=True, fancybox=True, framealpha=0.8, fontsize=8)
    plt.tight_layout()
    return fig
