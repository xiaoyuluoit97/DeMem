import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 读取平均值npy文件
file_path_1 = "base_average.npy"
file_path_2 = "DP_average.npy"

y_axis_2_file_path_1 = "base_tpracc_tpr01fpr_memwise.npy"
y_axis_2_file_path_2 = "DP_tpracc_tpr01fpr_memwise.npy"

# 读取数据
data1 = np.load(file_path_1)
data2 = np.load(file_path_2)
y_data1 = np.load(y_axis_2_file_path_1) * 0.01
y_data2 = np.load(y_axis_2_file_path_2) * 0.01

# 生成0-20的横坐标，并缩放到0-1
x = np.linspace(0, 1, len(data2))

# 设置字体大小
plt.rcParams.update({'font.size': 22})  # 统一调大字体

# 调整图表尺寸（使其更宽）
fig, ax1 = plt.subplots(figsize=(10, 8))

# 绘制左侧 y 轴的数据集，Base 圆形标记，DP 三角标记
ax1.plot(x, data1, marker='o', linestyle='-', color='tab:blue', label='Base')
ax1.plot(x, data2, marker='^', linestyle='--', color='tab:blue', label='DP')

# 设置左侧 y 轴的标签和颜色
ax1.set_xlabel('Memorization Score', fontsize=24,fontweight='bold')  # 增大横轴字体
ax1.set_ylabel('Natural accuracy',color='tab:blue', fontsize=24,fontweight='bold')  # 增大左侧y轴字体
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# 设置左侧 y 轴范围为 0 到 1
ax1.set_ylim(-0.03, 1)

# 创建右侧 y 轴，并绘制 Base 和 DP 的 TPR 数据，Base 圆形标记，DP 三角标记
ax2 = ax1.twinx()
ax2.plot(x, y_data1, marker='o', linestyle='-', color='tab:red', label='Base TPR')
ax2.plot(x, y_data2, marker='^', linestyle='--', color='tab:red', label='DP TPR')

# 设置右侧 y 轴的标签和颜色
ax2.set_ylabel('TPR @ 0.1% FPR', color='tab:red', fontsize=24,fontweight='bold')  # 增大右侧y轴字体
ax2.tick_params(axis='y', labelcolor='tab:red')

# 设置右侧 y 轴范围为 0 到 1
ax2.set_ylim(-0.03, 1)

# 设置标题并增大字体
#ax1.set_title('Utility and TPR Comparison', fontsize=16)

# 创建自定义图例，只显示实线和虚线，不显示颜色
legend_elements = [
    Line2D([0], [0],  marker='o', linestyle='-', color='gray', label='Base'),
    Line2D([0], [0], marker='^', linestyle='--', color='gray', label='DP-SGD')
]

# 添加图例并调整位置（这里将图例位置放到图表的右中间）
ax1.legend(handles=legend_elements, loc='center right', frameon=False)

# 保存图表为SVG格式
plt.savefig("combine.svg", format="svg")

# 显示图表
plt.show()
