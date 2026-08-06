"""
Reticle 综合分析（调用 batch 模块）
====================================
仅负责配置参数和调用批量分析器
功能：
1. 批量解析 PMUT Pickle 文件（Tx 或 Echo 均可）
2. 从文件名解析 wafer_id, row_id, die, C_index, 频率
3. 计算窗口 Vpp
4. 计算 -3dB 中心频率 (Fc) 和相对带宽 (RelBW) - 需进行 FFT 复数平均
5. 计算 Tx 灵敏度 (dB re 1 µPa/V) - 需水听器校准文件
6. 绘制 4 张图：
   - Vpp vs Reticle Column Index
   - Tx Sensitivity vs Reticle Column Index
   - -3dB Center Frequency vs Reticle Column Index
   - -3dB Relative Bandwidth vs Reticle Column Index
7. 保存 CSV 汇总文件
"""
import os
from batch.reticle_analyzer import run_reticle_analysis

# ==================== 配置参数 =====================
BASE_DIR = r'Z:\Measurements\2026\C1_Tx_and_txBW_260416\1cycle'
OUTPUT_DIR = os.path.join(BASE_DIR, 'Reticle_Combined_Analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    'response_window_start_us': 6.5,
    'response_window_end_us': 8.0,
    'fft_max_freq': 30e6,
    'fft_zero_padding': 4,
    'excitation_voltage_v': 60.0,
    'hydrophone_path': r'F:\MILKYWAY\SN4764_hydrophone_sensitivity.xlsx',
    'fig_formats': ['png'],
}

# ==================== 执行 =====================
if __name__ == "__main__":
    run_reticle_analysis(BASE_DIR, OUTPUT_DIR, CONFIG)


if __name__ == "__main__":
    main()
