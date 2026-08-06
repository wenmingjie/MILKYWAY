"""
PMUT Reticle 综合分析
========================================
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
import numpy as np
import pandas as pd
from collections import defaultdict

# 公共模块
from io.pickle_io import load_pickle_waveforms
from io.file_utils import parse_ids_from_filename, parse_frequency_from_filename
from analysis.waveform import remove_dc_offset, apply_time_window, compute_vpp
from analysis.spectrum import extract_bandwidth_metrics
from analysis.sensitivity import HydrophoneCalibration, compute_tx_sensitivity, tx_sensitivity_to_db_kpa_per_v
from plot.reticle import plot_grouped_metric_vs_column
from plot.style import set_style

# ==================== 配置参数 =====================
BASE_DIR = r'Z:\Measurements\2026\C1_Tx_and_txBW_260416\1cycle'   # 修改为实际路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'Reticle_Combined_Analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 响应时间窗（μs）
RESPONSE_WINDOW_START = 6.5
RESPONSE_WINDOW_END = 8.0

# FFT 参数（用于带宽计算）
FFT_MAX_FREQ = 30e6
FFT_ZERO_PADDING = 4

# 发射灵敏度参数
EXCITATION_VOLTAGE_V = 60.0   # 激励电压 (V)
HYDROPHONE_PATH = r'F:\MILKYWAY\SN4764_hydrophone_sensitivity.xlsx'

FIG_FORMATS = ['png']

# ==================== 主程序 =====================
def main():
    set_style()
    print("="*70)
    print("Reticle Combined Analysis: Vpp | Tx Sensitivity | Fc | RelBW")
    print("="*70)

    # 1. 初始化水听器校准
    hydro_cal = HydrophoneCalibration(HYDROPHONE_PATH)

    # 2. 扫描文件
    pickle_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.pickle')]
    if not pickle_files:
        print(f"错误: 在 {BASE_DIR} 中未找到 .pickle 文件")
        return
    print(f"发现 {len(pickle_files)} 个 pickle 文件")

    # 数据容器：按 (wafer, row, die) 分组，存储 (c_index, value)
    grouped_vpp = defaultdict(list)
    grouped_tx_db = defaultdict(list)
    grouped_fc = defaultdict(list)
    grouped_relbw = defaultdict(list)

    csv_rows = []

    # 3. 处理每个文件
    for idx, fname in enumerate(pickle_files, 1):
        file_path = os.path.join(BASE_DIR, fname)

        # 3a. 解析文件名
        freq_mhz = parse_frequency_from_filename(fname)
        if freq_mhz is None:
            print(f"[{idx}/{len(pickle_files)}] 跳过，无法解析频率: {fname}")
            continue

        wafer_id, reticle_id, row_id, c_index, die = parse_ids_from_filename(fname)
        if wafer_id is None:
            print(f"[{idx}/{len(pickle_files)}] 跳过，无法解析 ID: {fname}")
            continue

        # 3b. 加载信号
        signals_mV, time_us = load_pickle_waveforms(file_path, channel='ch2')
        if signals_mV is None:
            continue

        # 3c. 去直流
        signals_corr = remove_dc_offset(signals_mV, time_us, baseline_region=(0, 1))

        # 3d. 剔除第一次测量（若有多条）
        if signals_corr.shape[0] > 1:
            signals_corr = signals_corr[1:]
        else:
            print(f"[{idx}/{len(pickle_files)}] 警告: 仅 1 条波形，未剔除: {fname}")

        # 3e. 平均信号
        avg_signal = np.mean(signals_corr, axis=0)

        # 3f. 应用时间窗口
        win = (RESPONSE_WINDOW_START, RESPONSE_WINDOW_END)
        avg_win, time_win = apply_time_window(avg_signal[np.newaxis, :], time_us, win)
        avg_win = avg_win[0]
        all_win, _ = apply_time_window(signals_corr, time_us, win)

        # 3g. 计算 Vpp
        vpp = compute_vpp(avg_win)

        # 3h. 计算带宽指标（Fc 和 RelBW）
        peak_freq_mhz, fc_mhz, rel_bw = extract_bandwidth_metrics(
            all_win, time_win, FFT_MAX_FREQ, FFT_ZERO_PADDING, threshold='3dB'
        )

        # 3i. 计算 Tx 灵敏度（需要频率和水听器校准）
        hydro_sens = hydro_cal.get_sensitivity(freq_mhz)
        tx_sens = compute_tx_sensitivity(vpp, hydro_sens, EXCITATION_VOLTAGE_V)
        tx_sens_db = tx_sensitivity_to_db_kpa_per_v(tx_sens)

        # 3j. 存储分组数据
        key = (wafer_id, row_id, die)
        grouped_vpp[key].append((c_index, vpp))
        grouped_tx_db[key].append((c_index, tx_sens_db))
        grouped_fc[key].append((c_index, fc_mhz))
        grouped_relbw[key].append((c_index, rel_bw))

        # 3k. 准备 CSV 行
        csv_rows.append({
            'wafer_id': wafer_id,
            'row_id': row_id,
            'die': die,
            'reticle_id': reticle_id,
            'C_index': c_index,
            'Frequency_MHz': round(freq_mhz, 2),
            'Excitation_V': EXCITATION_VOLTAGE_V,
            'Response_Window_Start_us': RESPONSE_WINDOW_START,
            'Response_Window_End_us': RESPONSE_WINDOW_END,
            'Vpp_mV': round(vpp, 2),
            'Tx_Sensitivity_dB_re_1uPa_per_V': round(tx_sens_db, 2) if not np.isnan(tx_sens_db) else np.nan,
            'fc_3dB_MHz': round(fc_mhz, 2) if not np.isnan(fc_mhz) else np.nan,
            'relative_bw_percent': round(rel_bw, 2) if not np.isnan(rel_bw) else np.nan,
        })

        print(f"[{idx}/{len(pickle_files)}] {fname[:50]}... → "
              f"C{c_index}, Vpp={vpp:.2f} mV, Fc={fc_mhz:.2f} MHz, "
              f"Tx={tx_sens_db:.2f} dB")

    # 4. 保存 CSV
    df = pd.DataFrame(csv_rows)
    df.sort_values(['wafer_id', 'row_id', 'die', 'C_index'], inplace=True)
    csv_path = os.path.join(OUTPUT_DIR, 'reticle_combined_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV 已保存: {csv_path}")
    print(f"共处理 {len(csv_rows)} 条记录")

    # 5. 绘制 4 张图
    plot_grouped_metric_vs_column(
        grouped_vpp,
        ylabel='Received Voltage Vpp (mV)',
        title='Vpp vs Reticle Column Index',
        filename_prefix='Vpp_vs_Cindex',
        output_dir=OUTPUT_DIR,
        fig_formats=FIG_FORMATS
    )

    plot_grouped_metric_vs_column(
        grouped_tx_db,
        ylabel='Transmit Sensitivity (dB re 1 µPa/V)',
        title='PMUT Transmit Sensitivity vs Reticle Column Index',
        filename_prefix='Tx_Sensitivity_vs_Cindex',
        output_dir=OUTPUT_DIR,
        fig_formats=FIG_FORMATS
    )

    plot_grouped_metric_vs_column(
        grouped_fc,
        ylabel='-3 dB Center Frequency (MHz)',
        title='-3 dB Center Frequency vs Reticle Column Index',
        filename_prefix='fc_vs_Cindex',
        output_dir=OUTPUT_DIR,
        fig_formats=FIG_FORMATS
    )

    plot_grouped_metric_vs_column(
        grouped_relbw,
        ylabel='-3 dB Relative Bandwidth (%)',
        title='-3 dB Relative Bandwidth vs Reticle Column Index',
        filename_prefix='relative_bw_vs_Cindex',
        output_dir=OUTPUT_DIR,
        fig_formats=FIG_FORMATS
    )

    print("\n分析完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
