import os

import numpy as np
import torch
from dataset import get_cifar100_datasets
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc, roc_auc_score
from utils_h import computeMetrics
colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    'lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow', 'lightpink'
]
lam_to_color = {
    'mixuphu50': 'tab:blue',
    'mixupfilter': 'tab:orange',
    'mixupcut': 'tab:green',
    '80': 'tab:purple',
}
layer_to_color = {
    'layer1': 'tab:blue',
    'layer2': 'tab:orange',
    'layer3': 'tab:green',
    'layer4': 'tab:purple',
}
 # 定义颜色列表
root = "/data/luo/reproduce/privacy_and_aug"
root = "/Users/luo/reproduce/privacy_and_aug"
sample_num = 60000
class_labels = list(range(100))
LAYERANA = False
#MODELNUMBER = 'target'
MODELNUMBER = '64'
training_model = "resnet18"
with open("sampleinfo/samplelist.txt", "r") as f:
    samplelist = eval(f.read())

with open("sampleinfo/target.txt", "r") as f:
    samplelist_target = eval(f.read())

def normal(mu, sigma2, x):
    r = - np.log(sigma2) / 2 - (x - mu) ** 2 / (2 * sigma2)
    return r


def load_npy_files(directory,thr_type,mixuptyep,samplelist):
    sample_inds_dict = {}
    pred_list = []

    for filename in os.listdir(directory):
        parts = filename.split('_')
        file_content = {
            "prefix": parts[0],
            "thr": parts[1],
            "model": parts[2],
            "lam": parts[3].split('.')[0],
        }

        indicator = np.zeros(60000, dtype=np.bool_)
        #indicator[samplelist[int(file_content["model"])]] = True
        indicator[samplelist[int(64)]] = True
        #indicator[samplelist_target] = True
# and parts[5].split('.')[0] == str(51)
        if filename.endswith(".npy") and file_content["prefix"] == mixuptyep and file_content["thr"] == thr_type:
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            # Assuming your npy files contain both sample_inds and pred
            sample_inds_dict[filename] = data
            pred_list.append(indicator)

    return sample_inds_dict, pred_list


def ROC_curve(multiple_sample_inds, multiple_preds):
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    lam_colors = {}
    color_index = 0
    atktype = None
    thrvalue = None
    for sample_inds,pred in zip(multiple_sample_inds, multiple_preds.items()):
        parts = pred[0].split('_')
        file_content = {
            "prefix": parts[0],
            "thr": parts[1],
            "model": parts[2],
            "lam": parts[3].split('.')[0],  # Remove the .npy part
        }
        atktype = file_content["prefix"]
        thrvalue = file_content["thr"]
        lam = file_content["lam"]

        y = sample_inds

        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        pred_sort = np.sort(pred[1])[::-1]
        index = np.argsort(pred[1])[::-1]
        y_sort = y[index]
        tpr = []
        fpr = []
        thr = []
        for i, item in enumerate(pred_sort):
            tpr.append(np.sum((y_sort[:i] == 1)) / pos)
            fpr.append(np.sum((y_sort[:i] == 0)) / neg)
            thr.append(item)
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-3:
                tpr_0_1 = tpr[i] * 100
                break
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-4:
                tpr_00_1 = tpr[i] * 100
                break
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-5:
                tpr_000_1 = tpr[i] * 100
                break
        logfpr = np.log10(np.array(fpr) + 1e-5)
        logtpr = np.log10(np.array(tpr) + 1e-5) + 5
        eps = logfpr[1:] - logfpr[:-1]
        auroc_log = np.sum(eps * np.array(logtpr)[1:]) / (5 * 5)
        auroc = auc(fpr, tpr)

        fpr, tpr, thresholds = roc_curve(y, pred[1])
        AccList = 1 - np.logical_xor(pred[1][:, np.newaxis] > thresholds[np.newaxis, :], indicator[:, np.newaxis]).sum(0) / len(pred[1])
        Acc_opt = np.max(AccList)
        ind = np.argmax(AccList)
        thr_glo_opt = thresholds[ind]

        print("thr: " + str(thr_glo_opt))
        print("balance acc: " + str(Acc_opt))
        # Linear scale plot
        if LAYERANA:
            ax2.plot(fpr, tpr, label='layer: %s  AUROC %.2f tpr@0.1fpr: %.2f' % (file_content["model"], auroc_log, tpr_0_1), color=layer_to_color[file_content["model"]])
        else:
            ax2.plot(fpr, tpr, label=' LAM: %s  AUROC %.2f tpr@0.1fpr: %.2f' % (file_content["lam"],auroc, tpr_0_1),color=lam_to_color[lam])
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('True Positive Rate')
        ax2.set_xlabel('False Positive Rate')
        ax2.legend()


        # Log scale plot
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        if LAYERANA:
            ax1.plot(fpr, tpr, label='layer: %s  AUROC %.2f tpr@0.1fpr: %.2f' % (file_content["model"], auroc_log, tpr_0_1), color=layer_to_color[file_content["model"]])
        else:
            ax1.plot(fpr, tpr, label='lam: %s  AUROC %.2f tpr@0.1fpr: %.2f' % (file_content["lam"],auroc_log, tpr_0_1),color=lam_to_color[lam])
        ax1.plot([1e-7, 1], [1e-7, 1], 'r--')
        ax1.set_xlim([1e-7, 1])
        ax1.set_ylim([1e-7, 1])
        ax1.set_ylabel('True Positive Rate')
        ax1.set_xlabel('False Positive Rate')
        ax1.legend()

    #parts[5].split('.')[0] == str(51)
    #plt.title("LAM%s %s"% (parts[5].split('.')[0] == str(80),file_content["thr"]))
    plt.tight_layout()
    if LAYERANA:
        plt.savefig(f'Figure/{training_model}/feature{file_content["model"]}_{thrvalue}_{atktype}_roc.png')
    #plt.savefig(f'Figure/{training_model}/{thrvalue}_{atktype}_roc.png')
    else:
        plt.savefig(f'Figure/{training_model}/{thrvalue}_{atktype}_roc.png')
    plt.show()


def privacy_leak_each_class_model_mixup_lam(multiple_sample_inds, multiple_preds):
    fig, ax = plt.subplots(figsize=(10, 6))
    thrvalue = None
    atktype = None
    for color_index, (indicator, thr_value) in enumerate(zip(multiple_sample_inds, multiple_preds.items())):
        parts = thr_value[0].split('_')
        file_content = {
            "prefix": parts[0],
            "thr": parts[1],
            "model": parts[2],
            "lam": parts[3].split('.')[0]  # Remove the .npy part
        }
        atktype = file_content["prefix"]
        thrvalue = file_content["thr"]
        true_labels_per_class = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")

        y = np.zeros(sample_num)
        y[samplelist_target] = int(1)

        classes = np.unique(true_labels_per_class)

        class_roc_aucs = {}
        class_roc_aucs_global = {}
        class_accs = {}
        class_accs_global = {}
        class_tpr = {}
        class_tpr_global = {}

        fpr, tpr, thresholds = roc_curve(indicator, thr_value[1])
        AccList = 1 - np.logical_xor(thr_value[1][:, np.newaxis] > thresholds[np.newaxis, :], indicator[:, np.newaxis]).sum(0) / len(thr_value[1])
        ind = np.argmax(AccList)
        thr_opt = thresholds[ind]

        for class_label in classes:
            class_indices = np.where(true_labels_per_class == class_label)[0]
            class_phy = thr_value[1][class_indices]
            class_indicator = indicator[class_indices]

            fpr, tpr, thresholds = roc_curve(class_indicator, class_phy)
            #roc_auc = auc(fpr, tpr)
            #class_roc_aucs[class_label] = roc_auc

            AccList = 1 - np.logical_xor(class_phy[:, np.newaxis] > thresholds[np.newaxis, :],
                                         class_indicator[:, np.newaxis]).sum(0) / len(class_phy)
            Acc_opt = np.max(AccList)
            ind = np.argmax(AccList)
            #thr_cls_opt = thresholds[ind]
            tpr_opt = tpr[ind]
            class_tpr[class_label] = tpr_opt


            fpr_global, tpr_global, _ = roc_curve(class_indicator, thr_value[1][class_indices] > thr_opt)
            #roc_auc_global = auc(fpr_global, tpr_global)
            #class_roc_aucs_global[class_label] = roc_auc_global

            AccList_g = 1 - np.logical_xor(class_phy[:, np.newaxis] > thr_opt, class_indicator[:, np.newaxis]).sum(
                0) / len(class_phy)
            Acc_opt_g = np.max(AccList_g)

            class_tpr_global[class_label] = tpr_global[1]
            class_accs_global[class_label] = Acc_opt_g

        color = colors[color_index % len(colors)]
        if LAYERANA:
            ax.plot(classes, [class_tpr[class_label] for class_label in classes], label=f'mem opt - LAM: {file_content["lam"]}',linestyle='--', color=layer_to_color[file_content["model"]])
            ax.plot(classes, [class_tpr_global[class_label] for class_label in classes], label=f'global opt - LAM: {file_content["lam"]}', color=layer_to_color[file_content["model"]])
        else:
            ax.plot(classes, [class_tpr[class_label] for class_label in classes], label=f'mem opt - LAM: {file_content["lam"]}',linestyle='--', color=lam_to_color[file_content["lam"]])
            ax.plot(classes, [class_tpr_global[class_label] for class_label in classes], label=f'global opt - LAM: {file_content["lam"]}', color=lam_to_color[file_content["lam"]])
    ax.set_title(f'TPR rate of MIA, {thrvalue} {atktype}')
    ax.set_xlabel('memory value')
    ax.set_ylabel('TPR')
    ax.legend()
    ax.grid(True)

    plt.xticks(classes, range(len(classes)))
    plt.tight_layout()
    if LAYERANA:
        plt.savefig(f'Figure/{training_model}/{file_content["model"]}_{thrvalue}_{atktype}_tpr.png')
    #plt.savefig(f'Figure/{training_model}/{thrvalue}_{atktype}_roc.png')
    else:
        plt.savefig(f'Figure/{training_model}/{thrvalue}_{atktype}_tpr.png')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def draw_density(conf_file, indicators, bins=50):
    """
    绘制密度分布图。

    参数:
    conf_file (str): 包含conf值的.npy文件路径
    indicators (array-like): 包含True/False的indicator列表或数组
    bins (int): 直方图的分箱数目，默认值为50
    """
    # 从.npy文件中读取数据
    conf_values = np.load(conf_file)
    parts = conf_file.split('/')[1]
    parts = parts.split('_')
    file_content = {
        "prefix": parts[0],
        "thr": parts[1],
        "model": parts[2],
        "lam": parts[3].split('.')[0]  # Remove the .npy part
    }


    # 分割数据
    conf_member = conf_values[np.array(indicators) == True]
    conf_no_member = conf_values[np.array(indicators) == False]

    # 创建一个绘图区域
    plt.figure(figsize=(10, 6))

    # 绘制密度分布图
    plt.hist(conf_member, bins=bins, density=True, alpha=0.5, color='tab:blue', label='Member')
    plt.hist(conf_no_member, bins=bins, density=True, alpha=0.5, color='tab:orange', label='No Member')

    # 设置标签和标题
    plt.xlabel(f'{file_content["prefix"]} Value')
    plt.ylabel('Density')
    plt.title(f'{file_content["prefix"]} Density Distribution of {file_content["thr"]} Values')
    plt.legend()
    plt.savefig(f'Figure/{training_model}/{file_content["prefix"]}_{file_content["thr"]}_lam{file_content["lam"]}_density.png')
    # 显示图形
    plt.show()

# 示例用法
# draw_density('path_to_conf_file.npy', 'path_to_indicator_file.npy', bins=100)


# 示例用法
# draw_density('path_to_conf_file.npy', 'path_to_indicator_file.npy')










if __name__ == "__main__":

    directory = 'resnet18'
    indicator = np.zeros(60000, dtype=np.bool_)
    # indicator[samplelist[int(file_content["model"])]] = True
    #indicator[samplelist[int(0)]] = True
    indicator[samplelist_target] = True

    sample_inds_list, pred_list = load_npy_files(directory,"HUlossdiff","mixuphu",samplelist)
    ROC_curve(pred_list,sample_inds_list)
    #privacy_leak_each_class_model_mixup_lam(sample_inds_list,pred_list)
    #draw_density(os.path.join(directory,"mixuphu_conf_target_mixuphu.npy"),indicator,200)
    #pred_list, sample_inds_list

