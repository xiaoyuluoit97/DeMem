import numpy as np
import matplotlib.pyplot as plt



# 加载数据
data = np.load('sampleinfo/cifar100_infl_matrix.npz')
tr_mem = data['tr_mem']
data["tr_labels"]
diff_file = 'sampleinfo/loss_diff_target_65.npy'
tr_labels = data['tr_labels']
tt_labels = data['tt_labels']
gt_labels = np.concatenate((tr_labels, tt_labels))
# 假设有三个列表分别存储索引、类别标签和mem值
# 加载 indicator
with open("sampleinfo/target.txt", "r") as f:
    samplelist_target = eval(f.read())

indicator = np.zeros(60000, dtype=np.bool_)
indicator[samplelist_target] = True

# 加载diff_value2
diff_value2 = np.load(diff_file)

# 取前50000个点
tr_mem = tr_mem[:50000]
diff_value2 = diff_value2[:50000]

# 提取字段名称并去掉不需要的部分
diff_name = diff_file.split('/')[-1].split('.')[0]
diff_name = diff_name.replace('_target_65', '')

# 计算相关性
correlation = np.corrcoef(tr_mem, diff_value2)
print("Correlation coefficient:", correlation[0, 1])

# 随机选择5000个点
np.random.seed(120)  # 为了使结果可重复
sample_indices = np.random.choice(range(50000), size=5000, replace=False)
sample_tr_mem = tr_mem[sample_indices]
sample_diff_value2 = diff_value2[sample_indices]
sample_indicator = indicator[sample_indices]

# 根据indicator分组
group1_indices = sample_indicator == True
group2_indices = sample_indicator == False

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(sample_tr_mem[group1_indices], sample_diff_value2[group1_indices], alpha=0.5, color='blue', label='True')
plt.scatter(sample_tr_mem[group2_indices], sample_diff_value2[group2_indices], alpha=0.5, color='orange', label='False')
plt.title(f'Scatter Plot of tr_mem vs {diff_name} (Sampled 5000 Points)')
plt.xlabel('tr_mem')
plt.ylabel(diff_name)
plt.legend()
plt.grid(True)
plt.show()
