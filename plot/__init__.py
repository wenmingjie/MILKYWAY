from .style import set_style
from .waveform import plot_wave_with_window, plot_wave_overlay
from .spectrum import plot_spectrum_overlay, plot_avg_spectrum
from .figure import plot_time_freq_dual_axis_with_boxes
from .save import save_fig_all_formats

__all__ = [
    'set_style',
    'plot_wave_with_window',
    'plot_wave_overlay',
    'plot_spectrum_overlay',
    'plot_avg_spectrum',
    'plot_time_freq_dual_axis_with_boxes',
    'save_fig_all_formats',
]
