import numpy as np
import matplotlib.pyplot as plt

# 读取平均值npy文件
file_path_1 = "../双折现/base_average.npy"
file_path_2 = "../双折现/DP_average.npy"
file_path_3 = "pgdat_average.npy"
file_path_4 = "pgdat_reg_average.npy"

# 读取数据并缩小0.01倍
data1 = np.load(file_path_1)
data2 = np.load(file_path_2)
data3 = np.load(file_path_3)
data4 = np.load(file_path_4)

# 生成0-20的横坐标，并缩放到0-1
x = np.linspace(0, 1, len(data2))
plt.rcParams.update({'font.size': 14})

# 调整图表尺寸（使其更宽）
plt.figure(figsize=(8, 6))

# 绘制第一个数据集
plt.plot(x, data1, marker='o', linestyle='-', color='tab:blue', label='Base')

# 绘制第二个数据集
plt.plot(x, data2, marker='s', linestyle='-', color='tab:orange', label='DP')

# 绘制第三个数据集
#plt.plot(x, data3, marker='^', linestyle='-', color='tab:green', label='PGD')

# 绘制第四个数据集
#plt.plot(x, data4, marker='d', linestyle='-', color='tab:red', label='Trades')

# 添加标题和标签
plt.title('Utility')
plt.xlabel('Memorization Score')
plt.ylabel('Clean Sample Accuracy')

# 显示图例
plt.legend(loc='upper right')

# 显示网格
plt.grid(True)

# 保存图表为SVG格式
plt.savefig("pgd_utility.svg", format="svg")

# 显示图表
plt.show()
