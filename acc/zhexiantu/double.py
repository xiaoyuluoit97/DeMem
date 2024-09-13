import numpy as np
import matplotlib.pyplot as plt

# 定义数据
means_col1_tpr = [43.55, 50.91, 59.17, 67.23]
errors_col1_tpr = [1.18, 1.23, 1.58, 0.71]

means_col2_tpr = [36.14, 41.80, 52.55, 64.65]
errors_col2_tpr = [0.86, 1.81, 0.68, 0.69]

means_col1_rob_acc = [52.15, 41.98, 29.05, 15.78]
errors_col1_rob_acc = [0.25, 0.25, 0.12, 0.11]

means_col2_rob_acc = [50.70, 40.06, 28.43, 15.63]
errors_col2_rob_acc = [0.75, 0.70, 0.33, 0.14]

# 等距索引
x_values = np.arange(len(means_col1_rob_acc))  # [0, 1, 2, 3]
bar_width = 0.15  # 每个柱子的宽度
group_gap = 0.05   # 两组数据之间的间距

# 创建图像
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})

# 左侧 y 轴 (rob acc)
bars1 = ax1.bar(x_values - bar_width*2 - group_gap/2, means_col1_rob_acc, bar_width, yerr=errors_col1_rob_acc,
                label='PGD (Rob Acc)', color='tab:blue', capsize=5)
bars2 = ax1.bar(x_values - bar_width - group_gap/2, means_col2_rob_acc, bar_width, yerr=errors_col2_rob_acc,
                label='PGD+MEM (Rob Acc)', color='tab:cyan', capsize=5)
ax1.set_xlabel('$\epsilon$', fontsize=16)
ax1.set_ylabel('Robust Accuracy', fontsize=16, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(x_values)
ax1.set_xticklabels([1, 2, 4, 8])

# 右侧 y 轴 (TPR)
ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
bars3 = ax2.bar(x_values + bar_width + group_gap/2, means_col1_tpr, bar_width, yerr=errors_col1_tpr,
                label='PGD (TPR)', color='tab:red', capsize=5, alpha=0.7)
bars4 = ax2.bar(x_values + bar_width*2 + group_gap/2, means_col2_tpr, bar_width, yerr=errors_col2_tpr,
                label='PGD+MEM (TPR)', color='tab:orange', capsize=5, alpha=0.7)
ax2.set_ylabel('TPR @ 0.1% FPR', fontsize=16, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 添加标题
ax1.set_title('Privacy Leakage and Utility Comparison', fontsize=18)

# 在每个组上方添加数据标签，增加垂直偏移并使用白色背景以清晰可见
for i in range(len(x_values)):
    ax1.text(x_values[i] - bar_width*2 - group_gap/2, means_col1_rob_acc[i] + 1, 'PGD', ha='center', fontsize=12, color='tab:blue',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax1.text(x_values[i] - bar_width - group_gap/2, means_col2_rob_acc[i] + 1, 'PGD+MEM', ha='center', fontsize=12, color='tab:cyan',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax2.text(x_values[i] + bar_width + group_gap/2, means_col1_tpr[i] + 1, 'PGD', ha='center', fontsize=12, color='tab:red',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax2.text(x_values[i] + bar_width*2 + group_gap/2, means_col2_tpr[i] + 1, 'PGD+MEM', ha='center', fontsize=12, color='tab:orange',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# 调整布局
fig.tight_layout()

# 保存并显示图表
plt.savefig("diff_eps_dual_yaxis_bar_four_per_x_with_gap_and_clear_labels.svg", format="svg")
plt.show()
