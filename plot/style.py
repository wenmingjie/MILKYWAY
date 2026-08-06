import matplotlib.pyplot as plt

def set_style():
    """统一绘图风格"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.unicode_minus'] = False
    # 颜色可定义常量
    COLOR_TIME = '#0072BD'
    COLOR_FREQ = '#D95319'
    COLOR_THRESHOLD = 'orange'
    COLOR_PEAK = 'red'
    COLOR_BAND = 'purple'
