import numpy as np
import matplotlib.pyplot as plt

# 读取两个npy文件
#file_path_1 = "base/base_tpracc_tpr01fpr_memwise.npy"
file_path_4 = "pgdat_demem/pgdat_reg_8_0.6_tpracc_tpr01fpr_memwise.npy"
#file_path_2 = "dp/DP_tpracc_tpr01fpr_memwise.npy"
#file_path_2 = "trades/trades_tpracc_tpr01fpr_memwise.npy"
file_path_1 = "pgdat/pgdat_8_tpracc_tpr01fpr_memwise.npy"
#file_path_4 = "pgdat_demem/pgdat_reg_8_0.2_tpracc_tpr01fpr_memwise.npy"
data1 = np.load(file_path_1)* 0.01
data2 = np.load(file_path_4)* 0.01
#data3 = np.load(file_path_3)* 0.01
#data4 = np.load(file_path_4)* 0.01
minusval1 = data1 - data2
#minusval2 = data1 - data3

# 生成0-20的横坐标，并缩放到0-1
x = np.linspace(0, 1, len(data2))
plt.rcParams.update({'font.size': 14})

# 调整图表尺寸（使其更宽）
plt.figure(figsize=(8, 6))

# 绘制第一个数据集
#plt.plot(x, minusval1, marker='o', linestyle='-', color='tab:blue', label='The difference between base and DP')

# 绘制第二个数据集
plt.plot(x, data2, marker='s', linestyle='-', color='tab:orange', label='pgd_demem')

# 绘制第三个数据集
plt.plot(x, data1, marker='^', linestyle='-', color='tab:blue', label='pgd')

# 绘制第四个数据集
#plt.plot(x, data4, marker='d', linestyle='-', color='tab:red', label='Trades')

# 添加标题和标签
plt.title('The Privacy Leakage Under LiRA Attack')
plt.xlabel('Memorization Score')
plt.ylabel('TPR @ 0.1% FPR')

# 显示图例
plt.legend(loc='upper left')

# 显示网格
plt.grid(True)

plt.savefig("base_dp_diff_tpr01fpr.svg", format="svg")

# 显示图表
plt.show()
