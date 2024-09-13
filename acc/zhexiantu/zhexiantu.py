import numpy as np
import matplotlib.pyplot as plt

# 定义数据
means_col1_1 = [0.4355, 0.5091, 0.5917, 0.6723]

errors_col1_1 = [0.0118, 0.0123, 0.0158, 0.0071]

means_col2_1 = [0.3614, 0.4180, 0.5255, 0.6465]
errors_col2_1 = [0.0086, 0.0181, 0.0068, 0.0069]

means_col1_2 = [0.5215, 0.4198, 0.2905, 0.1578]
errors_col1_2 = [0.0025, 0.0025, 0.0012, 0.0011]

means_col2_2 = [0.5070, 0.4006, 0.2843, 0.1563]
errors_col2_2 = [0.0075, 0.0070, 0.0033, 0.0014]


# 等距索引
x_values = [1, 2, 3, 4]
plt.rcParams.update({'font.size': 18})
# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(12, 5))


# 第一个子图
axs[0].errorbar(x_values, means_col1_1, yerr=errors_col1_1, fmt='o-', capsize=5, label='PGD-AT')
axs[0].errorbar(x_values, means_col2_1, yerr=errors_col2_1, fmt='s-', capsize=5, label='PGD-AT + DeMem')
#axs[0].set_title('Privacy leakage risk')
axs[0].set_xlabel('$\epsilon$', fontsize=24,fontweight='bold')  # 调整x轴字体大小
axs[0].set_ylabel('TPR @ 0.1% FPR', fontsize=20,fontweight='bold')  # 调整y轴字体大小
axs[0].set_xticks(x_values)
axs[0].set_xticklabels([1, 2, 4, 8])
axs[0].legend()

# 第二个子图
axs[1].errorbar(x_values, means_col1_2, yerr=errors_col1_2, fmt='o-', capsize=5, label='PGD-AT')
axs[1].errorbar(x_values, means_col2_2, yerr=errors_col2_2, fmt='s-', capsize=5, label='PGD-AT + DeMem')
#axs[1].set_title('Robustness performance')
axs[1].set_xlabel('$\epsilon$', fontsize=24,fontweight='bold')  # 调整x轴字体大小
axs[1].set_ylabel('Robustness accuracy', fontsize=20,fontweight='bold')  # 调整y轴字体大小
axs[1].set_xticks(x_values)
axs[1].set_xticklabels([1, 2, 4, 8])
axs[1].legend()

# 调整布局
plt.tight_layout()

# 保存并显示图表
plt.savefig("diff_eps_subplot.svg", format="svg")
plt.show()
