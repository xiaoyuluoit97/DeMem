import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前目录下所有.npy文件
npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]

# 存储已处理的前缀
processed_prefixes = set()
point_size = 80
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))

# 存储所有数据以便排序
plot_data = []

for npy_file in npy_files:
    # 找到文件名中的前缀（即名称相同部分）
    alpha = npy_file.split('_')[-2]
    if alpha == "0" :
        continue
    prefix = '_'.join(npy_file.split('_')[:-1])

    # 如果这个前缀已经处理过，跳过
    if prefix in processed_prefixes:
        continue

    # 查找所有与当前文件前缀相同的文件
    matched_files = [f for f in npy_files if f.startswith(prefix)]

    # 如果找到了两对文件（x轴和y轴），进行绘图
    if len(matched_files) == 2:
        # 假设_xxx_robacc.npy作为x轴，_xxx_tpr01fpr.npy作为y轴
        if 'robacc' in matched_files[0] and 'tpr01fpr' in matched_files[1]:
            x_file, y_file = matched_files
        elif 'tpr01fpr' in matched_files[0] and 'robacc' in matched_files[1]:
            y_file, x_file = matched_files
        else:
            print(f"Skipping unmatched pair: {matched_files}")
            continue

        # 加载数据
        x_data = np.load(x_file)
        y_data = np.load(y_file)

        # 提取label
        label = npy_file.split('_')[-2]

        # 将数据存储在列表中
        plot_data.append((float(label), x_data, y_data, rf'$\lambda$ = {label}'))

        # 将前缀标记为已处理
        processed_prefixes.add(prefix)
    else:
        print(f"Found unmatched files for prefix '{prefix}': {matched_files}")

# 按label排序
plot_data.sort()

# 根据排序后的数据绘制散点图
for _, x_data, y_data, label in plot_data:
    plt.scatter(x_data, y_data, s=point_size, label=label)

# 添加标题和标签
plt.title('Utility vs. TPR @ 0.1% FPR')
plt.xlabel('Robust Accuracy')
plt.ylabel('TPR @ 0.1% FPR')

# 显示图例
plt.legend(loc='best')

# 显示网格
plt.grid(True)

# 保存图表为SVG格式
plt.savefig("reg_utility_plot.svg", format='svg')

# 显示图表
plt.show()
