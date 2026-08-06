import os
import subprocess
import shutil
import matplotlib.pyplot as plt

def save_fig_all_formats(base_path, fig=None, formats=None):
    """
    保存图像为指定格式 (png, pdf, emf)
    """
    if fig is None:
        fig = plt.gcf()
    if formats is None:
        formats = ['png']
    formats = [fmt.lower() for fmt in formats]
    
    png = base_path + ".png"
    pdf = base_path + ".pdf"
    emf = base_path + ".emf"
    
    if 'png' in formats:
        fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.3)
    if 'pdf' in formats or 'emf' in formats:
        fig.savefig(pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
    if 'emf' in formats:
        if os.path.exists(pdf) and shutil.which("inkscape") is not None:
            subprocess.run(["inkscape", pdf, "--export-type=emf", "--export-text-to-path",
                            f"--export-filename={emf}"], check=True)
            print(f"已保存: {emf}")
    # 关闭图形
    plt.close(fig)
