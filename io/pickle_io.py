import numpy as np
import pandas as pd
import os

def load_pickle_waveforms(file_path, channel='ch2', voltage_scale=1000):
    """
    从 Pickle 文件加载波形数据。
    返回: signals (mV), time_us (μs), 波形数量
    """
    try:
        data = pd.read_pickle(file_path)
        waveforms_df = data['waveforms']
        n_tests = len(waveforms_df)
        print(f"加载 {n_tests} 条波形: {os.path.basename(file_path)}")
        
        time_vector = np.array(waveforms_df['time'].iloc[0])
        time_us = time_vector * 1e6
        
        # 提取指定通道 (默认 ch2)
        signals_V = np.vstack([np.array(row[channel]) for _, row in waveforms_df.iterrows()])
        signals = signals_V * voltage_scale   # 转换为 mV 或保留 V，由调用者决定
        
        return signals, time_us, n_tests
    except Exception as e:
        print(f"错误: 加载 {file_path} 失败: {e}")
        return None, None, 0
