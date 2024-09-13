# from others_project.lixiao.privacy_and_aug.models import ResNet18
import os

import numpy as np
import torch
from dataset import get_cifar100_datasets
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    'lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow', 'lightpink'
]
print(colors)
 # 定义颜色列表
#root = "/data/luo/reproduce/privacy_and_aug"
root = "mia"
sample_num = 60000
class_labels = list(range(100))
MODELNUMBER = 88
MODELNUMBER = '0'
training_model = "resnet18"

with open("sampleinfo/samplelist.txt", "r") as f:
    samplelist = eval(f.read())

with open("sampleinfo/target.txt", "r") as f:
    samplelist_target = eval(f.read())

def normal(mu, sigma2, x):
    r = - np.log(sigma2) / 2 - (x - mu) ** 2 / (2 * sigma2)
    return r


def ROC_curve(sample_inds, pred, model,mixuptype,lam,save=False, show=True, name="test", aug_type=None):
    y = sample_inds
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]
    index = np.argsort(pred)[::-1]
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
    # Plotting
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Log scale plot
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.plot(fpr, tpr, 'k')
        ax1.set_title('AUROC %.2f lam: %s  tpr@0.1fpr: %.2f' % (auroc_log, str(lam), tpr_0_1))
        ax1.plot([1e-7, 1], [1e-7, 1], 'r--')
        ax1.set_xlim([1e-7, 1])
        ax1.set_ylim([1e-7, 1])
        ax1.set_ylabel('True Positive Rate')
        ax1.set_xlabel('False Positive Rate')

        # Linear scale plot
        ax2.plot(fpr, tpr, 'k')
        ax2.set_title('AUROC %.2f lam: %s  tpr@0.1fpr: %.2f' % (auroc, str(lam), tpr_0_1))
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('True Positive Rate')
        ax2.set_xlabel('False Positive Rate')

        plt.tight_layout()

        if save:
            plt.savefig(name + ".pdf", dpi=300)
        else:
            plt.savefig('Figure/' + model + '/' + mixuptype + lam + '.png')
            plt.show()

def calculate_percentage_change(base, comparison):

    if len(base) != len(comparison):
        raise ValueError("Both lists must have the same number of elements.")

    percentage_changes = []
    for base_val, comparison_val in zip(base, comparison):
        if base_val == 0:
            # Handle the case where the base value is 0
            change = 0 if comparison_val == 0 else float('inf')
        else:
            change = (comparison_val - base_val) / base_val * 100
        percentage_changes.append(change)

    return percentage_changes

# Testing the function with the provided lists
def plot_changes(list1,list2):
    plt.figure(figsize=(10,6))
    plt.plot(list1, marker='o', label='class-wise')
    plt.plot(list2, marker='^', label='global')
    #plt.plot(class_labels, pgd_org, marker='^', linestyle='--', color='tab:blue', label='pgdat-test')
    #plt.plot(class_labels, org_org, marker='^', linestyle='--', color='tab:orange', label='base-test')
    # 设置图的标题和标签
    plt.title('The privacy leakage of BASE to PGD model over different class')
    plt.xlabel('Class')
    plt.ylabel('Changes %')
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.show()



def plot_roc_curve(fpr,tpr,roc_auc,type,model,lam):

    for i in range(len(fpr) - 1, -1, -1):
        if fpr[i] <= 1e-3:
            #             print("TPR @ 0.1% FPR: ", (str(tpr[i] * 100)))
            tpr_0_1 = tpr[i] * 100
            break
    # Plot ROC curve
    print(f"tpr@0.1%fpr %s",tpr_0_1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC %s lam: %s  tpr@0.1fpr: %.2f' % (type,str(lam),tpr_0_1))
    plt.legend(loc="lower right")
    plt.savefig('Figure/' + model + '/'+type + lam + '.png')
    plt.show()



def privacy_leak_each_class_model():
    aug_types = ['base',"pgdat","AWP","trades","TradesAWP"]
    #thr_type_opt = {'LiRA':-0.108,'pgdat': 1.146, 'base': 3.262, 'AWP': 0.069, 'trades': -0.300,'TradesAWP': -0.850}
    #phy_diff : 5.117 64%  loss:5.746 62% 64% 7.553

    #thr_type_opt = {'pgdat':3.570,'base':5.161}
    #thr_type_opt = {'pgdat':1.146,'base':0.794}
    #loss
    #thr_type_opt = {'pgdat': 9.08, 'base': 9.851, 'AWP': 7.975, 'trades': 5.801,'TradesAWP': 5.169}
    dataset = 'cifar100'
    atktype = ('loss')
    aug_types = ["lira_0_pgdat_phy.npy", "lira_0_pgdatfat_phy.npy", "lira_0_trades_phy.npy", "lira_0_tradesfat_phy.npy"]

    class_accs_dict = {aug_type.split('_')[2]: [] for aug_type in aug_types}
    class_accs_global_dict = {aug_type.split('_')[2]: [] for aug_type in aug_types}
    class_tpr_dict = {aug_type.split('_')[2]: [] for aug_type in aug_types}
    class_tpr_global_dict = {aug_type.split('_')[2]: [] for aug_type in aug_types}
    agunum = 0
    aug_optimal_mia_acc = []


    for aug_type in aug_types:
        phy = np.load(os.path.join("temp/acc/mia",aug_type))
        #phy = np.load("sampleinfo/128model_baseeval_target0.npy")
        #target_phy = np.load("%s/%s_1.npy" % (dirs,atktype))
        aug_type = aug_type.split('_')[2]
        # 加载类别标签
        #true_labels_per_class = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")
        true_labels_per_class = np.load("sampleinfo/cifar100_train_labels.npy")

        # 假设 phy、indicator、sample_num 和 samplelist 已定义
        indicator = np.zeros(sample_num, dtype=np.bool_)
        #indicator_target = np.zeros(sample_num, dtype=np.bool_)
        #indicator[samplelist_target] = True
        indicator[samplelist[int(MODELNUMBER)]] = True

        # 分组数据并计算每个类别的 ROC 曲线和 AUC 值
        classes = np.unique(true_labels_per_class)

        class_thresholds = {}
        class_roc_curves = {}
        class_roc_aucs = {}
        class_accs = {}
        class_accs_global = {}
        class_roc_curves_global = {}
        class_roc_aucs_global = {}
        class_tpr = {}
        class_tpr_global = {}


        # 计算整体的 ROC AUC、准确率和最佳阈值
        fpr, tpr, thresholds = roc_curve(indicator, phy)
        roc_auc = auc(fpr, tpr)
        #plot_roc_curve(fpr, tpr,roc_auc,atktype,training_model,aug_type)
        AccList = 1 - np.logical_xor(phy[:, np.newaxis] > thresholds[np.newaxis, :], indicator[:, np.newaxis]).sum(0) / len(phy)
        Acc_opt = np.max(AccList)
        ind = np.argmax(AccList)
        thr_glo_opt = thresholds[ind]
        #AccList_target = 1 - np.logical_xor(target_phy[:, np.newaxis] > thr_opt, indicator_target[:, np.newaxis]).sum(0) / len(target_phy)
        # 输出整体的 ROC AUC、准确率和最佳阈值
        print("Overall: Acc = {:.3f}, Threshold = {:.3f}".format(Acc_opt, thr_glo_opt))
        aug_optimal_mia_acc.append(Acc_opt)

        print("Global opt的阈值和选用Class opt的阈值得到的TPR有负相关")
        #print(np.corrcoef(class_tpr_dict.get(aug_type), class_tpr_global_dict.get(aug_type)))


        for class_label in classes:
            class_indices = np.where(true_labels_per_class == class_label)[0]
            class_phy = phy[class_indices]
            class_indicator = indicator[class_indices]
            #target_class_indicator = indicator_target[class_indices]
            #target_class_phy = target_phy[class_indices]
            fpr, tpr, thresholds = roc_curve(class_indicator, class_phy)

            roc_auc = auc(fpr, tpr)

            class_roc_curves[class_label] = (fpr, tpr)
            class_roc_aucs[class_label] = roc_auc
            # 计算准确率和阈值
            AccList = 1 - np.logical_xor(class_phy[:, np.newaxis] > thresholds[np.newaxis, :],class_indicator[:, np.newaxis]).sum(0) / len(class_phy)
            Acc_opt = np.max(AccList)
            ind = np.argmax(AccList)
            thr_opt = thresholds[ind]
            tpr_opt = tpr[ind]
            class_tpr[class_label] = tpr_opt
            class_thresholds[class_label] = thr_opt

            _, tpr_cls, _ = roc_curve(class_indicator, phy[class_indices] > thresholds[ind])

            #target_accList = 1 - np.logical_xor(target_class_phy[:, np.newaxis] > thr_opt,target_class_indicator[:, np.newaxis]).sum(0) / len(target_class_phy)
            class_accs[class_label] = Acc_opt
            class_tpr_dict[aug_type].append(tpr_cls[1])
            class_accs_dict[aug_type].append(Acc_opt)

            # 使用全局最佳阈值计算 ROC 曲线
            fpr_global, tpr_global, _ = roc_curve(class_indicator, phy[class_indices] > thr_glo_opt)
            roc_auc_global = auc(fpr_global, tpr_global)
            class_roc_curves_global[class_label] = (fpr_global, tpr_global)
            class_roc_aucs_global[class_label] = roc_auc_global
            AccList_g = 1 - np.logical_xor(class_phy[:, np.newaxis] > thr_glo_opt, class_indicator[:, np.newaxis]).sum(0) / len(class_phy)
            Acc_opt_g = np.max(AccList_g)
            class_accs_global[class_label] = Acc_opt_g
            class_accs_global_dict[aug_type].append(Acc_opt_g)
            class_tpr_global_dict[aug_type].append(tpr_global[1])


        # 打印或使用每个类别的 ROC 曲线和 AUC 值
        #for (class_label, roc_auc),(class_label_g, roc_auc_g) in zip(class_roc_aucs.items(), class_roc_aucs_global.items()):
            #print("Class {}: Acc = {:.3f}, Threshold = {:.3f}".format(class_label,class_accs[class_label],class_thresholds[class_label]))

            #print("Class {}: Acc = {:.3f}, Threshold = {:.3f}".format(class_label_g,class_accs_global[class_label],thr_opt[aug_type]))




    plt.figure(figsize=(10, 6))
    for i, ((aug, accs), (aug_g, accs_same)) in enumerate(zip(class_tpr_dict.items(), class_tpr_global_dict.items())):
    #for i, ((aug, accs), (aug_g, accs_same)) in enumerate(zip(class_accs_dict.items(), class_accs_global_dict.items())):
        color = colors[i % len(colors)]  # 循环使用颜色列表中的颜色
        #plt.plot(classes, accs, marker='o', label=aug + '-class-opt-thr', color=color)
        #plt.plot(classes, accs_same, marker='^', linestyle='--', label=aug_g + '-glo-opt-thr', color=color)
        #plt.plot(classes, accs, linestyle='--',label=aug + '-class-opt-thr', color=color)
        plt.plot(classes, accs_same, label=aug_g + '-global-opt-thr', color=color)
        #plt.axhline(y=aug_optimal_mia_acc[i], linestyle=':', label=aug + '-overall', color=colors[i])
        np.save("acc/mia/%s_tpr_over_class.npy" % (aug_g), accs_same)

    # 设置图的标题和标签
    plt.title('TPR rate of %s MIA' % (atktype))
    plt.xlabel('memory value')
    plt.ylabel('TPR')
    #plt.ylim(0.54, 0.7)
    plt.xticks(classes, range(len(classes)))  # 设置 x 轴刻度
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.savefig('Figure/' + atktype + '.png')
    plt.show()

    #np.save('acc/mia/lira_pgdat_mem.npy', class_accs_dict.get('pgdat'))


if __name__ == '__main__':
    privacy_leak_each_class_model()
