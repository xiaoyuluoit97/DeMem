import numpy as np
from dataset import get_cifar100_datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
data = np.load('sampleinfo/cifar100_infl_matrix.npz')
traing_index_classidx_1 = data['tr_classidx_1']
testing_index_classidx_1 = data['tt_classidx_1']
# each datapoints label
# each test point label
tr_labels = data['tr_labels']
tt_labels = data['tt_labels']

tr_mem = data['tr_mem']
tr_mem = tr_mem


PRIVACY = False
def calculate_percentage_change(base, comparison):

    if len(base) != len(comparison):
        raise ValueError("Both lists must have the same number of elements.")

    percentage_changes = []
    for base_val, comparison_val in zip(base, comparison):
        if base_val == 0:
            # Handle the case where the base value is 0
            change = 0 if comparison_val == 0 else float('inf')
        else:
            change = (comparison_val - base_val) / base_val
        percentage_changes.append(change)

    return percentage_changes

def mem_bin_label():
    # 定义10个区间的边界
    bins = np.arange(0, 1.05, 0.05)  # [0, 0.1, 0.2, ..., 1.0]

    # 使用digitize为tr_mem中的每个值分配一个区间
    indices = np.digitize(tr_mem, bins, right=True)  # right=True表示区间是开闭的，即(0, 0.1]

    # 初始化一个字典来存储每个区间的索引
    interval_indices = {i: [] for i in range(0, 21)}

    # 分配每个mem值到相应的区间
    for idx, val in enumerate(indices):
        interval_indices[val].append(idx)

    # 现在interval_indices包含了10个区间的索引
    # 比如，interval_indices[1]将包含所有mem值在(0, 0.1]区间的训练样本的索引

    np.save("sampleinfo/cifar100_mem_21bin_labels.npy", indices)
    print()

def writeit():
    train,test = get_cifar100_datasets(device='cpu', use_aug = False, multiple_query = False)
    assert torch.allclose(train.labels, test.labels)
    # 提取类别标签
    labels_array = train.labels.numpy()

    # 保存 NumPy 数组为 .npy 文件
    np.save("sampleinfo/cifar100_train_labels.npy", labels_array)

    print("Train labels have been saved to train_labels.npy file.")
    return labels_array

class_mem_stats = {}
#writeit()
def mem_gt():
    for K in range(0, 100):  # 对于CIFAR-100的每个类别
        # 构建索引数组的名称
        class_idx_var_name = f'tr_classidx_{K}'

        # 获取当前类别的训练样本索引
        class_idx = data[class_idx_var_name]  # 假设data是之前加载的npz文件内容

        # 获取当前类别的mem值
        class_mem_values = tr_mem[class_idx]

        # 计算平均值和方差
        mean_value = np.mean(class_mem_values)
        variance_value = np.var(class_mem_values)

        # 存储结果
        class_mem_stats[K] = {'mean': mean_value, 'variance': variance_value}

    mean_values = np.array([class_mem_stats[k]['mean'] for k in range(0, 100)])
    variance_values = np.array([class_mem_stats[k]['variance'] for k in range(0, 100)])
    return mean_values, variance_values




CLASS_WISE = False
if __name__ == "__main__":
    if CLASS_WISE:
        # mean,var = mem_gt()
        class_mem = np.load("sampleinfo/cifar100_class_mem.npy")
        sorted_indices = np.argsort(class_mem)
        sorted_class_mem = class_mem[sorted_indices]

        # rob_acc = np.load("acc/pgd-fat-clean/pgdat_rob_acc_overclass.npy")
        # mia_acc = np.load("acc/mia/lira_pgdat_class.npy")
        # print(mean)
        # mia_acc_files = ["conf_base_class.npy", "conf_pgdat_class.npy","conf_DP_class.npy","conf_trades_class.npy"]
        # mia_acc_files = ["DP_average_acc_class.npy","base_average_acc_class.npy","pgdat_average_acc_class.npy","trades_average_acc_class.npy","pgdat_fat_average_acc_class.npy","trades_fat_average_acc_class.npy"]

        # mia_acc_files = ["pgdat_average_rob_acc_class.npy","trades_average_rob_acc_class.npy", "FATpgdat_average_rob_acc_class.npy","FATtrades_average_rob_acc_class.npy"]
        # mia_acc_files =["pgdat_tpr_over_class.npy","pgdatfat_tpr_over_class.npy","trades_tpr_over_class.npy","tradesfat_tpr_over_class.npy"]
        # mia_acc_files = ["DP_privacy_TPR_acc_class.npy","conf_privacy_TPR_acc_class.npy"]

        # mia_acc_files = ["pgdat_privacy_TPR_classwise.npy", "FATpgdat_privacy_TPR_classwise.npy"]
        # mia_acc_files = ["trades_privacy_TPR_classwise.npy", "FATtrades_privacy_TPR_classwise.npy"]
        # mia_acc_data = [np.load(f"acc/mia/PRIVACY/{file}") for file in mia_acc_files]
        # base_path = "/Users/xiaoyuluo/workspace/privacy_and_aug-main/acc/mia/PRIVACY/result"
        base_path = "/Users/xiaoyuluo/workspace/privacy_and_aug-main/acc/tpr/dp"
        mia_acc_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]
        # 使用列表推导式从新的路径加载文件
        mia_acc_data = [np.load(f"{base_path}/{file}") for file in mia_acc_files]
        # 对每个数据集进行排序
        sorted_mia_acc_data = [data[sorted_indices] for data in mia_acc_data]

        # 设置图表大小
        plt.figure(figsize=(12, 6))

        # 绘制每个排序后的数据集
        for i, sorted_mia_acc in enumerate(sorted_mia_acc_data):
            plt.plot(range(len(sorted_mia_acc)), sorted_mia_acc, marker='o', label=mia_acc_files[i].split("_")[0])

        # 设置标题、标签和图例的字体大小
        title_fontsize = 18  # 标题字体大小
        label_fontsize = 16  # 标签字体大小
        legend_fontsize = 14  # 图例字体大小，可以根据需要调整

        if PRIVACY:
            plt.title('Class-wise MIA accuracy (privacy)', fontsize=title_fontsize)
            plt.xticks(ticks=range(len(sorted_mia_acc_data[0])), labels=sorted_indices, rotation='vertical', fontsize=6)
            plt.xlabel('Class Index (Sorted by class average memorization value)', fontsize=label_fontsize)
            plt.ylabel('TPR', fontsize=label_fontsize)

        else:
            plt.title('Class-wise robustness accuracy (utility)')
            plt.xticks(ticks=range(len(sorted_mia_acc_data[0])), labels=sorted_indices, rotation='vertical',
                       fontsize=6)  # 调整fontsize的值以适应你的需求

            plt.xlabel('Class Index (Sorted by class average memorization value)')
            plt.ylabel('Accuracy')

        plt.grid(True, which='both', axis='y')
        plt.legend(fontsize=legend_fontsize)
        plt.savefig('icassp_figure/base_dp_at_clean_acc.svg', format='svg')
        plt.show()



    #correlation_matrix = np.corrcoef(class_mem, mia_acc)
    #plot_line(clean_rob_acc, miaacc_pgdat_glothr*100)
    # correlation_matrix是一个2x2矩阵，其中对角线元素是自相关，非对角线元素是list1和list2之间的相关系数
    #correlation_coefficient = correlation_matrix[0, 1]
    #print(correlation_coefficient)
