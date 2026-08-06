import matplotlib.pyplot as plt
import numpy as np
from .style import set_style
from .save import save_fig_all_formats

def plot_grouped_metric_vs_column(
    grouped_data,          # dict: (wafer, row, die) -> list of (c_index, y_value)
    ylabel,
    title,
    filename_prefix,
    output_dir,
    fig_formats=['png'],
    column_label='Reticle Column Index (C)',
    marker_map=None,
    cmap_name='tab10'
):
    """
    绘制分组散点连线图，每个组显示一条线，并标出平均值虚线。
    grouped_data: 字典，键为 (wafer_id, row_id, die)，值为列表 (c_index, y_value)
    """
    set_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = plt.get_cmap(cmap_name)
    wafer_color = {}
    def get_color(w):
        if w not in wafer_color:
            wafer_color[w] = cmap(len(wafer_color) % 10)
        return wafer_color[w]

    if marker_map is None:
        marker_map = {
            'R0': '<', 'R1': '^', 'R2': 's', 'R3': 'o',
            'R4': 'D', 'R5': 'v', 'R6': '>'
        }
    default_marker = 'x'

    legend_handles = []

    for (wafer, row, die), data in sorted(grouped_data.items()):
        data_sorted = sorted(data, key=lambda x: x[0])
        c_indices = [d[0] for d in data_sorted]
        y_values = [d[1] for d in data_sorted]
        mean_val = np.nanmean(y_values)
        std_val = np.nanstd(y_values)
        cv = std_val / mean_val * 100 if mean_val != 0 else np.nan

        color = get_color(wafer)
        marker = marker_map.get(row, default_marker)

        # 连线
        ax.plot(c_indices, y_values, color=color, lw=1.2, alpha=0.85)
        # 散点
        ax.scatter(c_indices, y_values, s=70, marker=marker,
                   color=color, edgecolors='black', zorder=3)
        # 平均值虚线
        ax.plot([min(c_indices), max(c_indices)],
                [mean_val, mean_val], linestyle='--',
                color=color, lw=1.2, alpha=0.7)

        # 图例标签
        label = f'{wafer}_{row}Cx_{die}\nμ = {mean_val:.2f}'
        if cv is not None and not np.isnan(cv):
            label += f', CV = {cv:.2f}%'
        legend_handles.append(
            plt.Line2D([0], [0], color=color, marker=marker,
                       linestyle='-', lw=1.2, markersize=7, label=label)
        )

    # 设置横坐标
    all_c = sorted({ci for data in grouped_data.values() for ci, _ in data})
    ax.set_xticks(all_c)
    ax.set_xticklabels([f'C{ci}' for ci in all_c])

    ax.set_xlabel(column_label, fontsize=13.5)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=11, title='Wafer_Row_Die')
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename_prefix)
    save_fig_all_formats(save_path, fig=fig, formats=fig_formats)
    return fig
