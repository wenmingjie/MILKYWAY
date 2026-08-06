"""
CSV 文件夹时频分析（调用 batch 模块）
======================================
仅负责配置参数并调用批量分析器
功能：
- 从两个文件夹分别加载CSV（频域/时域）
- 计算时域参数、复数平均频谱、带宽、HD2
- 绘制双轴叠加图（含信息框）
"""
import os
from batch.csv_folder_analyzer import CSVBatchAnalyzer

# ==================== 配置参数 =====================
CONFIG = {
    'folder_path': r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz',
    'die_id': 'STB3SP11W00_R3C0_L',
    'window_freq': (6.5, 8.0),     # 频域窗口 (μs)
    'window_time': (6.5, 8.0),     # 时域窗口 (μs)
    'fft_max_freq': 30e6,
    'fft_zero_padding': 4,
    'output_dir': r'Y:\Measurements\2026\L_sweep_Tx对称性_20260708\1cycle-sine-20MHz',
    'title_prefix': 'Transmit Response Waveform',  # 可选，默认同上
}

# ==================== 执行 =====================
if __name__ == "__main__":
    analyzer = CSVBatchAnalyzer(CONFIG)
    results = analyzer.run()
