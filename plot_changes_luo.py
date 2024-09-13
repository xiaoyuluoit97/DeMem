import os

import numpy as np
import torch

import mem
from dataset import get_cifar10_datasets
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc, roc_auc_score

def plot_line(list1,list2):
    plt.figure(figsize=(10,6))
    plt.plot(list1, marker='o', label='pgd')
    plt.plot(list2, marker='^', label='global')

    #plt.plot(class_labels, pgd_org, marker='^', linestyle='--', color='tab:blue', label='pgdat-test')
    #plt.plot(class_labels, org_org, marker='^', linestyle='--', color='tab:orange', label='base-test')

    # 设置图的标题和标签
    plt.title('The acc gap from clean to clean on PGD model over different class')
    plt.xlabel('Class')
    plt.ylabel('Changes %')
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.show()


def load_npy_file(file_path):
    """
    读取.npy文件并返回其中的内容

    参数:
    file_path (str): .npy 文件的路径

    返回:
    np.ndarray: 包含.npy文件内容的 NumPy 数组
    """
    # 使用 NumPy 的 load 函数加载.npy文件
    data = np.load(file_path)

    return data


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

def calculate_abs_change(base, comparison):

    if len(base) != len(comparison):
        raise ValueError("Both lists must have the same number of elements.")

    abs_changes = []
    for base_val, comparison_val in zip(base, comparison):
        if base_val == 0:
            # Handle the case where the base value is 0
            change = 0 if comparison_val == 0 else float('inf')
        else:
            change = comparison_val - base_val
        abs_changes.append(change)

    return abs_changes

if __name__ == '__main__':

    base_clean = load_npy_file('acc/cifar100/base_clean_acc.npy')
    pgdat_clean = load_npy_file('acc/cifar100/pgdat_clean_acc.npy')
    pgdat_rob = load_npy_file('acc/cifar100/pgdat_rob_acc.npy')
    awp_clean = load_npy_file('acc/cifar100/AWP_clean_acc.npy')
    awp_rob = load_npy_file('acc/cifar100/AWP_rob_acc.npy')
    trades_clean = load_npy_file('acc/cifar100/trades_clean_acc.npy')
    trades_rob = load_npy_file('acc/cifar100/trades_rob_acc.npy')
    tradesAWP_clean = load_npy_file('acc/cifar100/TradesAWP_clean_acc.npy')
    tradesAWP_rob = load_npy_file('acc/cifar100/TradesAWP_rob_acc.npy')

    miatpr_base_clsthr = load_npy_file('privacyacc/cifar100/mia/miatpr_base_clsthr.npy')
    miatpr_base_glothr = load_npy_file('privacyacc/cifar100/mia/miatpr_base_glothr.npy')


    #miaacc_pgdat_clsthr = load_npy_file('privacyacc/cifar100/mia/miaacc_pgdat_clsthr.npy')
    #miaacc_base_clsthr = load_npy_file('privacyacc/cifar100/mia/miaacc_base_clsthr.npy')
    #miaacc_pgdat_glothr = load_npy_file('privacyacc/cifar100/mia/miaacc_pgdat_glothr.npy')
    #miaacc_base_glothr = load_npy_file('privacyacc/cifar100/mia/miaacc_base_glothr.npy')



    pgd_bias = calculate_percentage_change(pgdat_clean, pgdat_rob)
    awp_bias = calculate_percentage_change(awp_clean, awp_rob)
    trades_bias = calculate_percentage_change(trades_clean, trades_rob)
    tradesAWP_bias = calculate_percentage_change(tradesAWP_clean, tradesAWP_rob)
    trades_abs_bias = calculate_abs_change(trades_clean, trades_rob)

    privacy_decrease_from_base_to_pgd_byclsthr = load_npy_file(
        'privacyacc/cifar100/privacy_decrease_from_base_to_pgd_byclsthr.npy')
    privacy_decrease_from_base_to_pgd_byglothr = load_npy_file(
        'privacyacc/cifar100/privacy_decrease_from_base_to_pgd_byglothr.npy')

    privacy_decrease_fromthr_glo_clas_pgd = load_npy_file(
        'privacyacc/cifar100/privacy_decrease_fromthr_glo_clas_pgd.npy')
    privacy_decrease_fromthr_glo_clas_base = load_npy_file(
        'privacyacc/cifar100/privacy_decrease_fromthr_glo_clas_base.npy')

    mem_avg,mem_var = mem.mem_gt()


    print(miatpr_base_clsthr)
    print(miatpr_base_glothr)
    correlation_matrix = np.corrcoef(miatpr_base_glothr,miatpr_base_clsthr)
    #plot_line(clean_rob_acc, miaacc_pgdat_glothr*100)
    # correlation_matrix是一个2x2矩阵，其中对角线元素是自相关，非对角线元素是list1和list2之间的相关系数
    correlation_coefficient = correlation_matrix[0, 1]
    print(correlation_coefficient)