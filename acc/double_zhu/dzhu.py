import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 数据1
lambdas_1 = [0.2, 0.4, 0.6]
rob_acc_1 = [0.1563, 0.1464, 0.1350]
rob_acc_err_1 = [0.0014, 0.0015, 0.0027]
tpr_1 = [0.6465, 0.6012, 0.5514]
tpr_err_1 = [0.0069, 0.0064, 0.0098]

# 数据2
lambdas_2 = [0.2, 0.3, 0.4]
rob_acc_2 = [0.1703, 0.1677, 0.1518]
rob_acc_err_2 = [0.0029, 0.0031, 0.0031]
tpr_2 = [0.4969, 0.4780, 0.4252]
tpr_err_2 = [0.0142, 0.0088, 0.0110]

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # figsize 控制图像的总宽度和高度
plt.rcParams.update({'font.size': 18})
# 设置柱状图宽度和位置偏移
bar_width = 0.20
bar_offset = 0.15
index_1 = np.arange(len(lambdas_1))
index_2 = np.arange(len(lambdas_2))

# 左侧子图（数据1）
bars1 = ax1.bar(index_1 - bar_offset, rob_acc_1, bar_width, yerr=rob_acc_err_1, color='tab:blue', label='rob acc', capsize=5)
ax1.set_xlabel(r'$\lambda$', fontsize=20)
ax1.set_ylabel('Robust accuracy', color='tab:blue', fontsize=15,fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16, width=2)
ax1.tick_params(axis='x', labelsize=16, width=2)
ax1.set_ylim([0, 0.2])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_title('PGD-AT', fontsize=20,fontweight='bold')

# 设置左侧x轴的刻度为0.2, 0.4, 0.6
ax1.set_xticks(index_1)
ax1.set_xticklabels(lambdas_1)

ax1_twin = ax1.twinx()  # 创建第二个y轴
bars1_twin = ax1_twin.bar(index_1 + bar_offset, tpr_1, bar_width, yerr=tpr_err_1, color='tab:orange', label='TPR@0.1%FPR', capsize=5)
ax1_twin.set_ylabel('TPR @ 0.1% FPR', color='tab:orange', fontsize=17,fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='tab:orange', labelsize=16, width=2)
ax1_twin.set_ylim([0.5, 0.7])
ax1_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# 右侧子图（数据2）
bars2 = ax2.bar(index_2 - bar_offset, rob_acc_2, bar_width, yerr=rob_acc_err_2, color='tab:blue', label='rob acc', capsize=5)
ax2.set_xlabel(r'$\lambda$', fontsize=20,fontweight='bold')
ax2.set_ylabel('Robust accuracy', color='tab:blue', fontsize=17,fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=16, width=2)
ax2.tick_params(axis='x', labelsize=16, width=2)
ax2.set_ylim([0, 0.2])
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_title('TRADES', fontsize=20,fontweight='bold')

# 设置右侧x轴的刻度为0.2, 0.3, 0.4
ax2.set_xticks(index_2)
ax2.set_xticklabels(lambdas_2)

ax2_twin = ax2.twinx()  # 创建第二个y轴
bars2_twin = ax2_twin.bar(index_2 + bar_offset, tpr_2, bar_width, yerr=tpr_err_2, color='tab:orange', label='TPR@0.1%FPR', capsize=5)
ax2_twin.set_ylabel('TPR @ 0.1% FPR', color='tab:orange', fontsize=15,fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor='tab:orange', labelsize=14, width=2)
ax2_twin.set_ylim([0.35, 0.55])
ax2_twin.set_yticks([0.35, 0.40, 0.45, 0.50, 0.55])
ax2_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# 调整布局和保存图片
fig.tight_layout()
plt.savefig("all_reg_tpr_rob_subplot.svg", format="svg")
plt.show()
