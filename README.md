# MILKYWAY 项目结构

MILKYWAY 是用于 PMUT/MEMS 声学测试的数据分析平台。

项目采用模块化设计，将数据读取、分析、绘图、批处理、文档等功能解耦，方便代码复用、版本管理和后续功能扩展。

---

## 根目录

### README.md

项目说明文档。

包括项目介绍、目录结构、使用方法及开发规范。

---

### CHANGELOG.md

版本更新记录。

记录每次重要修改，例如：

- 新增功能
- 算法调整
- Bug 修复
- 绘图风格更新

---

### requirements.txt

Python 环境依赖。

例如：

- numpy
- scipy
- pandas
- matplotlib
- openpyxl

方便快速搭建运行环境。

---

### .gitignore

Git 忽略文件配置。

避免将临时文件、缓存、测试数据等上传到仓库。

---

# docs/

项目文档。

用于记录算法、流程、设计思路，而不是代码。

建议长期维护。

## Algorithms/

算法说明。

记录每种算法的：

- 原理
- 公式
- 参数说明
- 使用场景
- 注意事项

例如：

```
FFT.md
Bandwidth.md
Gaussian.md
```

以后忘记算法时，不需要翻代码。

---

## Workflow/

分析流程文档。

记录各种测试的数据处理流程。

例如：

```
Echo.md

Sweep.md

Noise.md
```

内容包括：

- 输入数据
- 分析流程
- 输出结果
- 使用脚本

方便以后快速复用。

---

## Examples/

使用示例。

介绍各模块如何调用。

例如：

- 如何读取 pickle
- 如何计算 FFT
- 如何绘制时域+频域图

---

# io/

数据输入输出模块。

只负责文件读写，不进行任何数据分析。

建议放：

```
pickle_io.py

csv_io.py

excel_io.py

export.py
```

主要功能：

- 读取 pickle
- 读取 csv
- 保存 csv
- 保存 Excel
- 导出分析结果

---

# analysis/

数据分析模块。

整个项目的核心。

所有算法均放在这里。

每个模块负责一类分析。

## waveform.py

时域分析。

建议放：

- Peak
- Vpp
- RMS
- 包络
- 到达时间
- 时域窗口处理

---

## spectrum.py

频域分析。

建议放：

- FFT
- 峰值频率
- 中心频率
- -3dB 带宽
- -6dB 带宽
- Relative BW
- HD2
- HD3
- Noise Floor

所有频谱相关分析统一放在这里。

---

## sweep.py

Sweep 数据分析。

例如：

- 最大响应
- 中心频率
- Sweep 带宽
- Tx Sensitivity
- 曲线拟合

---

## noise.py

噪声分析。

例如：

- PSD
- RMS Noise
- Noise Floor
- SNR

---

## statistics.py

统计分析。

例如：

- 平均值
- 标准差
- 最大值
- 最小值
- Batch 汇总
- 分组统计

---

# plot/

绘图模块。

统一管理所有图形。

所有分析脚本均调用这里进行绘图。

避免每个脚本重复写 matplotlib。

---

## waveform.py

时域图。

例如：

- 单波形
- 平均波形
- 多波形比较

---

## spectrum.py

频域图。

例如：

- FFT
- 多 FFT 比较
- 频谱标注

---

## figure.py

组合图。

负责多个子图布局。

例如：

- Time + FFT
- 多窗口排版
- 论文图布局

注意：

这里只负责布局，不负责分析。

---

## sweep.py

Sweep 曲线绘制。

例如：

- Frequency Response
- Tx Sensitivity
- Bandwidth Curve

---

## wafer.py

Wafer Map 绘图。

例如：

- Reticle 分布
- Die 分布
- Heatmap

---

## style.py

统一绘图风格。

建议统一管理：

- 字体
- 字号
- 颜色
- DPI
- LineWidth
- Figure Size

修改一次即可影响所有绘图。

---

# batch/

批量处理模块。

用于处理多个文件。

主要负责：

- 遍历文件夹
- 调用分析模块
- 调用绘图模块
- 保存结果

尽量不要在这里编写具体算法。

例如：

```
batch_echo.py

batch_sweep.py

batch_noise.py
```

---

# examples/

示例程序。

也是整个项目真正运行的入口。

这里的脚本尽量保持简单。

主要负责：

1. 读取数据
2. 调用 analysis
3. 调用 plot
4. 导出结果

例如：

```
echo_analysis.py

noise_analysis.py

wafer_analysis.py
```

以后新增测试类型，只需要增加新的 Example。

---

# archive/

历史版本归档。

用于保存：

- 旧版本脚本
- 已淘汰算法
- 历史项目

避免影响当前代码。

原则：

Archive 内代码只保存，不再继续开发。

---

# validation/

算法验证数据。

用于验证算法修改前后的结果是否一致。

建议保存：

- 标准测试数据（Golden Data）
- 已验证的数据集
- 对应的参考结果

例如：

```
Echo/

Sweep/

Noise/
```

以后修改算法时，可重新运行验证数据，确认：

- FFT 是否一致
- Bandwidth 是否变化
- Vpp 是否变化
- HD2 是否变化

保证算法更新不会影响已有分析结果。
