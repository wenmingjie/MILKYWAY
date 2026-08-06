import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class HydrophoneCalibration:
    """水听器校准类，加载灵敏度曲线并插值"""
    def __init__(self, excel_path, freq_col='Freq(MHz)', sens_col='Sensitivity(mV/Mpa)'):
        df = pd.read_excel(excel_path)
        self.freq = df[freq_col].values
        self.sens = df[sens_col].values
        self.interp = interp1d(self.freq, self.sens, kind='cubic', fill_value="extrapolate")

    def get_sensitivity(self, freq_mhz):
        return float(self.interp(freq_mhz))

def compute_tx_sensitivity(vpp_mV, hydro_sens_mV_per_MPa, excitation_V):
    """
    计算发射灵敏度 (kPa/V)
    vpp_mV: 水听器接收到的 Vpp (mV)
    hydro_sens_mV_per_MPa: 水听器灵敏度 (mV/MPa)
    excitation_V: 激励电压 (V)
    """
    if vpp_mV <= 0 or hydro_sens_mV_per_MPa <= 0 or excitation_V <= 0:
        return np.nan
    return vpp_mV * 1000 / (hydro_sens_mV_per_MPa * excitation_V)

def tx_sensitivity_to_db_kpa_per_v(sens_kpa_per_v):
    """将 kPa/V 转换为 dB re 1 µPa/V"""
    if sens_kpa_per_v <= 0 or np.isnan(sens_kpa_per_v):
        return np.nan
    return 20 * np.log10(sens_kpa_per_v) + 180
