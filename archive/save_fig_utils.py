# -*- coding: utf-8 -*-
"""
save_fig_utils.py
图像保存与格式转换工具：
- 支持 PNG、PDF、EMF 格式任意组合保存
- 使用 Inkscape 将 PDF 转换为 EMF（文字转路径）
- 自动控制字体、中文兼容、显示/关闭图形
"""

import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === PDF 输出配置 ===
rcParams['pdf.fonttype'] = 42               # 保留文本信息
rcParams['pdf.use14corefonts'] = False
rcParams['text.usetex'] = False
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 中英文兼容字体
plt.rcParams['axes.unicode_minus'] = False              # 解决负号显示问题


def check_inkscape():
    """检测 Inkscape 是否可用"""
    if shutil.which("inkscape") is None:
        print("未检测到 Inkscape，请确保已安装并添加到系统 PATH。")
        return False
    return True


def convert_pdf_to_emf(pdf_file, emf_file):
    """使用 Inkscape 将 PDF 转换为 EMF"""
    if not os.path.exists(pdf_file):
        print(f"错误：PDF 文件不存在 -> {pdf_file}")
        return False

    if not check_inkscape():
        return False

    try:
        result = subprocess.run([
            "inkscape",
            pdf_file,
            "--export-type=emf",
            "--export-text-to-path",
            f"--export-filename={emf_file}"
        ], check=True, capture_output=True, text=True)

        if os.path.exists(emf_file):
            print(f"转换成功：{emf_file}")
            return True
        else:
            print(f"转换失败：未生成 EMF 文件")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Inkscape 执行失败：{e.stderr}")
        return False

def save_fig_all_formats(base_path, fig=None, show_plots=False, formats=None):
    """
    保存图像为指定格式（可多选）
    参数：
        base_path : 不含扩展名的路径，如 "F:/results/plot1"
        fig : matplotlib Figure 对象（可选）
        show_plots : 是否显示图像（默认 False）
        formats : 要保存的格式列表，例如：
                  ['png'], ['pdf'], ['emf'], ['png','pdf'], ['png','pdf','emf']
                  默认保存 ['png','pdf','emf']
    """
    if fig is None:
        fig = plt.gcf()

    if formats is None:
        formats = ['png', 'pdf', 'emf']

    formats = [fmt.lower() for fmt in formats]

    png = base_path + ".png"
    pdf = base_path + ".pdf"
    emf = base_path + ".emf"

    # --- PNG ---
    if 'png' in formats:
        fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.3)
        # print(f"已保存: {png}")

    # --- PDF ---
    if 'pdf' in formats or 'emf' in formats:
        fig.savefig(pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
        # print(f"已保存: {pdf}")

    # --- EMF ---
    if 'emf' in formats:
        if os.path.exists(pdf):
            if convert_pdf_to_emf(pdf, emf):
                print(f"已保存: {emf}")
            else:
                print(f"EMF 转换失败: {emf}")
        else:
            print(f"PDF 文件未生成，无法转换为 EMF")

    # --- 显示或关闭 ---
    # 修改开始：移除 plt.close(fig)，只保留显示逻辑
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
