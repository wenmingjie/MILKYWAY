import matplotlib.pyplot as plt
from .style import set_style, COLOR_TIME, COLOR_PEAK

def plot_wave_with_window(time_us, signal_avg, window_start=None, window_end=None,
                           title=None, xlabel='Time (μs)', ylabel='Voltage (mV)',
                           figsize=(10,6), vpp=None):
    """
    绘制平均波形，可选显示窗口垂直线。
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_us, signal_avg, color=COLOR_TIME, lw=1.2)
    ax.set_xlabel(xlabel, fontsize=13.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    legend_handles = [plt.Line2D([0],[0], color=COLOR_TIME, lw=1.2, label='Average Signal')]
    if vpp is not None:
        legend_handles.append(plt.Line2D([0],[0], color='white', lw=0, label=f'Vpp: {vpp:.2f} mV'))
    if window_start is not None and window_end is not None:
        ax.axvline(window_start, color='red', linestyle='--', alpha=0.8)
        ax.axvline(window_end, color='red', linestyle='--', alpha=0.8)
        legend_handles.append(plt.Line2D([0],[0], color='red', linestyle='--', alpha=0.8,
                                         label=f'Window({window_start:.1f}-{window_end:.1f} μs)'))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02,1),
              frameon=True, fancybox=True, framealpha=0.8, fontsize=11)
    plt.tight_layout()
    return fig

def plot_wave_overlay(time_us, signals, vpp_list, max_traces=20, window_us=None,
                      title=None, xlabel='Time (μs)', ylabel='Voltage (mV)',
                      figsize=(10,6)):
    """
    绘制多条波形叠加，并显示每条波形的 Vpp 统计信息。
    signals: (n, N)
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    n = min(max_traces, signals.shape[0])
    for i in range(n):
        ax.plot(time_us, signals[i], lw=1.0, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=13.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_title(title if title else 'Time-Domain Waveforms Overlay', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    legend_elements = []
    for i in range(n):
        legend_elements.append(plt.Line2D([0],[0], color=ax.lines[i].get_color(), lw=1.0,
                                          label=f'Trace {i+1}: Vpp={vpp_list[i]:.2f} mV'))
    legend_elements.append(plt.Line2D([0],[0], color='white', lw=0,
                                      label=f'Total: {signals.shape[0]} traces'))
    legend_elements.append(plt.Line2D([0],[0], color='white', lw=0,
                                      label=f'Avg Vpp: {np.mean(vpp_list):.2f} mV'))
    legend_elements.append(plt.Line2D([0],[0], color='white', lw=0,
                                      label=f'Std Vpp: {np.std(vpp_list):.2f} mV'))
    if window_us is not None:
        legend_elements.append(plt.Line2D([0],[0], color='white', lw=0,
                                          label=f'Window: {window_us[0]:.1f}-{window_us[1]:.1f} μs'))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02,1),
              frameon=True, fancybox=True, framealpha=0.8, fontsize=9)
    plt.tight_layout()
    return fig
