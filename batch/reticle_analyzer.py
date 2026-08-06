"""
Reticle 批量分析器
===================
负责：
1. 扫描指定文件夹中的所有 pickle 文件
2. 对每个文件执行分析（Vpp, Tx 灵敏度, 带宽等）
3. 按 (wafer, row, die) 分组汇总
4. 导出 CSV
5. 生成分组图（Vpp, Tx, Fc, RelBW）
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


class ReticleBatchAnalyzer:
    """
    Reticle 批量分析器
    封装所有批量处理逻辑，简化示例脚本
    """

    def __init__(self, base_dir, output_dir, config):
        """
        参数:
            base_dir: 包含 pickle 文件的根目录
            output_dir: 输出目录
            config: dict，包含以下键：
                - response_window_start_us: 窗口起始 (μs)
                - response_window_end_us: 窗口结束 (μs)
                - fft_max_freq: FFT 最大频率 (Hz)
                - fft_zero_padding: FFT 零填充倍数
                - excitation_voltage_v: 激励电压 (V)
                - hydrophone_path: 水听器校准 Excel 路径
                - fig_formats: 图片格式列表
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.config = config

        # 初始化水听器校准
        self.hydro_cal = HydrophoneCalibration(config['hydrophone_path'])

        # 数据容器
        self.grouped_vpp = defaultdict(list)
        self.grouped_tx_db = defaultdict(list)
        self.grouped_fc = defaultdict(list)
        self.grouped_relbw = defaultdict(list)
        self.csv_rows = []

    def analyze_file(self, file_path):
        """
        分析单个 pickle 文件，返回解析后的指标
        若解析失败返回 None
        """
        fname = os.path.basename(file_path)

        # 1. 解析频率
        freq_mhz = parse_frequency_from_filename(fname)
        if freq_mhz is None:
            return None

        # 2. 解析 ID
        wafer_id, reticle_id, row_id, c_index, die = parse_ids_from_filename(fname)
        if wafer_id is None:
            return None

        # 3. 加载信号
        signals_mV, time_us = load_pickle_waveforms(file_path, channel='ch2')
        if signals_mV is None:
            return None

        # 4. 去直流
        signals_corr = remove_dc_offset(signals_mV, time_us, baseline_region=(0, 1))

        # 5. 剔除第一次测量
        if signals_corr.shape[0] > 1:
            signals_corr = signals_corr[1:]

        # 6. 平均信号
        avg_signal = np.mean(signals_corr, axis=0)

        # 7. 应用窗口
        win = (self.config['response_window_start_us'],
               self.config['response_window_end_us'])
        avg_win, time_win = apply_time_window(avg_signal[np.newaxis, :], time_us, win)
        avg_win = avg_win[0]
        all_win, _ = apply_time_window(signals_corr, time_us, win)

        # 8. 计算 Vpp
        vpp = compute_vpp(avg_win)

        # 9. 计算带宽指标
        peak_freq_mhz, fc_mhz, rel_bw = extract_bandwidth_metrics(
            all_win, time_win,
            self.config['fft_max_freq'],
            self.config['fft_zero_padding'],
            threshold='3dB'
        )

        # 10. 计算 Tx 灵敏度
        hydro_sens = self.hydro_cal.get_sensitivity(freq_mhz)
        tx_sens = compute_tx_sensitivity(vpp, hydro_sens, self.config['excitation_voltage_v'])
        tx_sens_db = tx_sensitivity_to_db_kpa_per_v(tx_sens)

        # 11. 返回结果
        return {
            'fname': fname,
            'wafer_id': wafer_id,
            'row_id': row_id,
            'die': die,
            'reticle_id': reticle_id,
            'c_index': c_index,
            'freq_mhz': freq_mhz,
            'vpp': vpp,
            'tx_sens_db': tx_sens_db,
            'peak_freq_mhz': peak_freq_mhz,
            'fc_mhz': fc_mhz,
            'rel_bw': rel_bw,
        }

    def run(self):
        """执行批量分析"""
        set_style()
        print("="*70)
        print("Reticle Batch Analyzer: Vpp | Tx Sensitivity | Fc | RelBW")
        print("="*70)

        # 扫描文件
        pickle_files = [f for f in os.listdir(self.base_dir) if f.endswith('.pickle')]
        if not pickle_files:
            print(f"错误: 在 {self.base_dir} 中未找到 .pickle 文件")
            return

        print(f"发现 {len(pickle_files)} 个 pickle 文件")

        # 逐个分析
        for idx, fname in enumerate(pickle_files, 1):
            file_path = os.path.join(self.base_dir, fname)
            result = self.analyze_file(file_path)

            if result is None:
                print(f"[{idx}/{len(pickle_files)}] 跳过: {fname}")
                continue

            # 存储分组数据
            key = (result['wafer_id'], result['row_id'], result['die'])
            self.grouped_vpp[key].append((result['c_index'], result['vpp']))
            self.grouped_tx_db[key].append((result['c_index'], result['tx_sens_db']))
            self.grouped_fc[key].append((result['c_index'], result['fc_mhz']))
            self.grouped_relbw[key].append((result['c_index'], result['rel_bw']))

            # 存储 CSV 行
            self.csv_rows.append({
                'wafer_id': result['wafer_id'],
                'row_id': result['row_id'],
                'die': result['die'],
                'reticle_id': result['reticle_id'],
                'C_index': result['c_index'],
                'Frequency_MHz': round(result['freq_mhz'], 2),
                'Excitation_V': self.config['excitation_voltage_v'],
                'Response_Window_Start_us': self.config['response_window_start_us'],
                'Response_Window_End_us': self.config['response_window_end_us'],
                'Vpp_mV': round(result['vpp'], 2),
                'Tx_Sensitivity_dB_re_1uPa_per_V': (
                    round(result['tx_sens_db'], 2)
                    if not np.isnan(result['tx_sens_db']) else np.nan
                ),
                'fc_3dB_MHz': (
                    round(result['fc_mhz'], 2)
                    if not np.isnan(result['fc_mhz']) else np.nan
                ),
                'relative_bw_percent': (
                    round(result['rel_bw'], 2)
                    if not np.isnan(result['rel_bw']) else np.nan
                ),
            })

            print(f"[{idx}/{len(pickle_files)}] {fname[:40]}... → "
                  f"C{result['c_index']}, Vpp={result['vpp']:.2f} mV, "
                  f"Fc={result['fc_mhz']:.2f} MHz")

        # 保存 CSV
        self.save_csv()
        # 生成所有图表
        self.plot_all()

    def save_csv(self):
        """保存 CSV 汇总"""
        df = pd.DataFrame(self.csv_rows)
        df.sort_values(['wafer_id', 'row_id', 'die', 'C_index'], inplace=True)
        csv_path = os.path.join(self.output_dir, 'reticle_combined_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nCSV 已保存: {csv_path}")
        print(f"共处理 {len(self.csv_rows)} 条记录")

    def plot_all(self):
        """生成所有分组图"""
        fig_formats = self.config.get('fig_formats', ['png'])

        plots = [
            (self.grouped_vpp, 'Received Voltage Vpp (mV)',
             'Vpp vs Reticle Column Index', 'Vpp_vs_Cindex'),
            (self.grouped_tx_db, 'Transmit Sensitivity (dB re 1 µPa/V)',
             'PMUT Transmit Sensitivity vs Reticle Column Index', 'Tx_Sensitivity_vs_Cindex'),
            (self.grouped_fc, '-3 dB Center Frequency (MHz)',
             '-3 dB Center Frequency vs Reticle Column Index', 'fc_vs_Cindex'),
            (self.grouped_relbw, '-3 dB Relative Bandwidth (%)',
             '-3 dB Relative Bandwidth vs Reticle Column Index', 'relative_bw_vs_Cindex'),
        ]

        for data, ylabel, title, prefix in plots:
            if data:  # 仅在数据非空时绘制
                plot_grouped_metric_vs_column(
                    data,
                    ylabel=ylabel,
                    title=title,
                    filename_prefix=prefix,
                    output_dir=self.output_dir,
                    fig_formats=fig_formats
                )

        print("\n所有图表已生成")


# ==================== 便捷函数 ====================
def run_reticle_analysis(base_dir, output_dir, config):
    """便捷函数：创建分析器并运行"""
    analyzer = ReticleBatchAnalyzer(base_dir, output_dir, config)
    analyzer.run()
    return analyzer
