import os
import csv
from datetime import datetime

def append_results_to_csv(output_dir, filename, results_dict):
    """
    将 results_dict 追加到 CSV 文件，若文件不存在则创建并写入表头。
    """
    csv_path = os.path.join(output_dir, filename)
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(results_dict)
    print(f"结果已保存到: {csv_path}")
    return csv_path
