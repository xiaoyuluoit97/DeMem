# from others_project.lixiao.privacy_and_aug.models import ResNet18
import os

import numpy as np
import torch
from dataset import get_cifar100_datasets
from matplotlib import pyplot as plt
import random
from sklearn.metrics import roc_curve,auc, roc_auc_score
from utils_h import computeMetrics
colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    'lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow', 'lightpink'
]

print(colors)
 # 定义颜色列表
#root = "/data/luo/reproduce/privacy_and_aug"


root = "/Users/xiaoyuluo/workspace/privacy_and_aug-main/mia"
root = "/Users/xiaoyuluo/workspace/privacy_and_aug-main/acc/mia/"
root = "/home/luo/acc/mia/"
sample_num = 60000

class_labels = list(range(100))

training_model = "resnet18"
with open("sampleinfo/samplelist.txt", "r") as f:
    samplelist = eval(f.read())

with open("sampleinfo/target.txt", "r") as f:
    samplelist_target = eval(f.read())

lira = True

class_mem = np.load("sampleinfo/cifar100_class_mem.npy")
sort = np.argsort(class_mem)
sorted_class_mem = class_mem[sort]
DATASET = "svhn"

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

def compute_feature(aug_type, test_ids=0):
    dirs = os.path.join(root, "phy", 'cifar10', aug_type)
    atktype = 'phy'
    allmodellist = list(range(0, 4))
    allmodellist.remove(0)

    IN  = []
    OUT = []

    for i in range(sample_num):
        IN.append([])
        OUT.append([])
    alls = [i for i in range(sample_num)]
    for index in allmodellist:
        slist = samplelist[index]
        for it in slist:
            IN[it].append(index)
        outslist = set(alls) - set(slist)
        for it in outslist:
            OUT[it].append(index)

    npdict = dict()
    for index in allmodellist:
        #npdict[index] = np.load("%s/phy_%s.npy" % (dirs, str(index)))[:, test_ids]
        #npdict[index] = np.load("%s/phy_%s.npy" % (dirs, str(index)))
        npdict[index] = np.load("%s/%s_%s.npy" % (dirs, atktype,str(index)))

    print('computing mean & var for in & out')
    in_dict = dict()
    out_dict = dict()
    confs_dict = dict()
    for i in range(sample_num):
        confsin, confsout = [], []
        for it in IN[i]:
            x = npdict[it][i]
            confsin.append(x)
        for it in OUT[i]:
            x = npdict[it][i]
            confsout.append(x)
        confsin = np.array(confsin)
        confsout = np.array(confsout)
        in_u, in_sigma = np.mean(confsin), np.var(confsin)
        out_u, out_sigma = np.mean(confsout), np.var(confsout)
        in_dict[i] = (in_u, in_sigma)
        out_dict[i] = (out_u, out_sigma)
        confs_dict[i] = (confsin, confsout)
    # mean  /  variance
    print('test one model')
    # base = np.load("%s/phy_%s.npy" % (dirs, str(trial)))[:, test_ids]
    base = np.load("%s/%s_0.npy" % (dirs,atktype))
    base_in = []
    base_out = []
    baseeval_offline = []
    for i in range(sample_num):
        a = normal(in_dict[i][0], in_dict[i][1], base[i])
        b = normal(out_dict[i][0], out_dict[i][1], base[i])
        base_in.append(a)
        base_out.append(b)
        baseeval_offline.append((base[i] - out_dict[i][0]) / np.sqrt(out_dict[i][1]))
    base_in = np.array(base_in)
    base_out = np.array(base_out)
    baseeval = base_in - base_out
    baseeval_offline = np.array(baseeval_offline)
    return base, baseeval, confs_dict
print('done')

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

def plot_acc_bias():

    basemodel_cleandataset = [0.9469, 0.9607, 0.8925, 0.8482, 0.9298, 0.8772, 0.9537, 0.9408, 0.9661, 0.9675]
    basemodel_pgddataset = [0.0, 0.0215, 0.0029, 0.0, 0.0, 0.0367, 0.0066, 0.0134, 0.0636, 0.0603]
    pgdmodel_cleandataset = [0.8753, 0.9328, 0.7149, 0.6719, 0.7898, 0.7016, 0.8813, 0.8697, 0.9321, 0.9122]
    pgdmodel_pgddataset = [0.5939, 0.7136, 0.3476, 0.2419, 0.3339, 0.3625, 0.5589, 0.6219, 0.6797, 0.6422]

    plot_changes(calculate_percentage_change(basemodel_cleandataset,pgdmodel_cleandataset), calculate_percentage_change(pgdmodel_cleandataset,pgdmodel_pgddataset))

    plt.figure()
    class_labels = [0,1,2,3,4,5,6,7,8,9]
    plt.plot(class_labels, basemodel_cleandataset, marker='o', label='base-clean',color=colors[0])
    plt.plot(class_labels, basemodel_pgddataset, marker='^', linestyle='--', label='base-clean',color=colors[0])
    plt.plot(class_labels, pgdmodel_cleandataset, marker='o', label='pgd-clean',color=colors[1])
    plt.plot(class_labels, pgdmodel_pgddataset, marker='^', linestyle='--', label='pgd-clean',color=colors[1])
    plt.grid(True)  # 显示网格线
    plt.xlabel('Class')
    plt.ylabel('Predict ACC')
    plt.title('Predict Accuracy Per Class')
    plt.legend()
    plt.yticks([i / 5 for i in range(6)])
    plt.show()

def writeit():
    train,test = get_cifar100_datasets(device='cpu', use_aug = False, multiple_query = False)
    assert torch.allclose(train.labels, test.labels)
    # 提取类别标签
    labels_array = train.labels.numpy()

    # 保存 NumPy 数组为 .npy 文件
    np.save("sampleinfo/cifar100_mem21bin_labels.npy", labels_array)

    print("Train labels have been saved to train_labels.npy file.")


from sklearn.metrics import roc_curve, auc
import numpy as np


def privacy_leak_each_mem_model():
    aug_types = ['base',"pgdat","trades","DP"]
    #,
    #thr_type_opt = {'LiRA':-0.108,'pgdat': 1.146, 'base': 3.262, 'AWP': 0.069, 'trades': -0.300,'TradesAWP': -0.850}
    #phy_diff : 5.117 64%  loss:5.746 62% 64% 7.553

    #thr_type_opt = {'pgdat':3.570,'base':5.161}
    #thr_type_opt = {'pgdat':1.146,'base':0.794}
    #loss
    #thr_type_opt = {'pgdat': 9.08, 'base': 9.851, 'AWP': 7.975, 'trades': 5.801,'TradesAWP': 5.169}
    dataset = 'cifar100'
    atktype = ('conf')
    class_accs_dict = {aug_type: [] for aug_type in aug_types}
    class_accs_global_dict = {aug_type: [] for aug_type in aug_types}
    class_tpr_dict = {aug_type: [] for aug_type in aug_types}
    class_tpr_global_dict = {aug_type: [] for aug_type in aug_types}
    agunum = 0
    aug_optimal_mia_acc = []

    for aug_type in aug_types:
        print("Augmentation type: {}".format(aug_type))

        dirs = os.path.join(root, aug_type)

        if aug_type == "DP":
            print("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
            phy = np.load("%s/%s_%s.npy" % (dirs,atktype,0))
        else:
            print("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
            phy = np.load("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
        #phy = np.load("sampleinfo/128model_baseeval_target0.npy")
        #target_phy = np.load("%s/%s_1.npy" % (dirs,atktype))

        # 加载类别标签
        true_labels_per_class = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")
        #true_labels_per_class = np.load("sampleinfo/cifar100_train_labels.npy")

        # 假设 phy、indicator、sample_num 和 samplelist 已定义
        indicator = np.zeros(sample_num, dtype=np.bool_)
        #indicator_target = np.zeros(sample_num, dtype=np.bool_)
        indicator[samplelist_target] = True
        #indicator[samplelist[int(MODELNUMBER)]] = True

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
        print(np.corrcoef(class_tpr_dict.get(aug_type), class_tpr_global_dict.get(aug_type)))



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
        #plt.plot(classes, accs_same, label=aug_g + '-global-opt-thr', color=color)
        #plt.axhline(y=aug_optimal_mia_acc[i], linestyle=':', label=aug + '-overall', color=colors[i])




    #np.save('privacyacc/cifar100/mia/miatpr_base_clsthr.npy', class_tpr_dict.get('pgdat'))
    #np.save('privacyacc/cifar100/mia/miatpr_base_glothr.npy', class_tpr_global_dict.get('pgdat'))

    #plt.plot(class_labels, pgd_org, marker='^', linestyle='--', color='tab:blue', label='pgdat-test')
    #plt.plot(class_labels, org_org, marker='^', linestyle='--', color='tab:orange', label='base-test')

    # 设置图的标题和标签
    plt.title('TPR rate of %s MIA' % (atktype))
    plt.xlabel('memory value')
    plt.ylabel('TPR')
    #plt.ylim(0.54, 0.7)
    plt.xticks(classes, range(len(classes)))  # 设置 x 轴刻度
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.savefig('Figure/' + atktype + '.png')
    #plt.show()


    np.save('privacyacc/cifar100/mia/lira_class', class_accs_dict.get('pgdat'))


def privacy_leak_each_class_model(MODELNUMBER,CLASS_WISE,aug_types):
    #aug_types = ['base',"pgdat","trades","DP"]
    #thr_type_opt = {'LiRA':-0.108,'pgdat': 1.146, 'base': 3.262, 'AWP': 0.069, 'trades': -0.300,'TradesAWP': -0.850}
    #phy_diff : 5.117 64%  loss:5.746 62% 64% 7.553
    #thr_type_opt = {'pgdat':3.570,'base':5.161}
    #thr_type_opt = {'pgdat':1.146,'base':0.794}
    #loss
    #thr_type_opt = {'pgdat': 9.08, 'base': 9.851, 'AWP': 7.975, 'trades': 5.801,'TradesAWP': 5.169}
    dataset = 'cifar100'
    atktype = ('phy')
    class_accs_dict = {aug_type: [] for aug_type in aug_types}
    class_accs_global_dict = {aug_type: [] for aug_type in aug_types}
    class_tpr_dict = {aug_type: [] for aug_type in aug_types}
    class_tpr_global_dict = {aug_type: [] for aug_type in aug_types}
    class_tprat01fpr_global_dict = {aug_type: [] for aug_type in aug_types}
    class_tprat0001fpr_global_dict = {aug_type: [] for aug_type in aug_types}
    agunum = 0
    aug_optimal_mia_acc = []

    for aug_type in aug_types:
        print("Augmentation type: {}".format(aug_type))

        dirs = os.path.join(root,DATASET,aug_type)

        if aug_type == "DP" and lira:
            if BEST_MODEL:
                phy = np.load("%s/lira_num%s_DP_best_phy.npy" % (dirs, MODELNUMBER))
            else:
                phy = np.load("%s/lira_num%s_DP_phy.npy" % (dirs, MODELNUMBER))
            #phy = np.load("%s/conf_%s.npy" % (dirs, MODELNUMBER))

        elif aug_type == "DP" :
            if BEST_MODEL:
                phy = np.load("%s/%s_best_%s.npy" % (dirs, atktype, MODELNUMBER))
            else:
                print("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
                phy = np.load("%s/%s_%s.npy" % (dirs,atktype,MODELNUMBER))

        elif lira:
            if BEST_MODEL:
                phy = np.load("%s/lira_num%s_%s_best_phy.npy" % (dirs, MODELNUMBER, aug_type))
            else:
                phy = np.load("%s/lira_num%s_%s_phy.npy" % (dirs,MODELNUMBER,aug_type))

        else:
            print("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
            phy = np.load("%s/%s_%s.npy" % (dirs, atktype, MODELNUMBER))
        #phy = np.load("sampleinfo/128model_baseeval_target0.npy")
        #target_phy = np.load("%s/%s_1.npy" % (dirs,atktype))

        # 加载类别标签
        if CLASS_WISE:
            true_labels_per_class = np.load("sampleinfo/cifar100_train_labels.npy")
        else:
            true_labels_per_class = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")

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
        tpr_glo = tpr
        fpr_glo = fpr
        roc_auc = auc(fpr, tpr)
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-3:
                tpr_0_1_class_glo_opt = tpr[i] * 100
                break
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-4:
                tpr_00_1_class_glo_opt = tpr[i] * 100
                break
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 1e-5:
                tpr_000_1_class_glo_opt = tpr[i] * 100
                break
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= 0:
                tpr_0_0_class_glo_opt = tpr[i] * 100
                break
        #plot_roc_curve(fpr, tpr,roc_auc,atktype,training_model,aug_type)
        AccList = 1 - np.logical_xor(phy[:, np.newaxis] > thresholds[np.newaxis, :], indicator[:, np.newaxis]).sum(0) / len(phy)
        Acc_opt_glo = np.max(AccList)
        ind = np.argmax(AccList)
        thr_glo_opt = thresholds[ind]
        tpr_glo_opt = tpr[ind]
        #AccList_target = 1 - np.logical_xor(target_phy[:, np.newaxis] > thr_opt, indicator_target[:, np.newaxis]).sum(0) / len(target_phy)
        # 输出整体的 ROC AUC、准确率和最佳阈值
        print("Overall: Acc = {:.3f}, Threshold = {:.3f}".format(Acc_opt_glo, thr_glo_opt))
        aug_optimal_mia_acc.append(Acc_opt_glo)

        print("Global opt的阈值和选用Class opt的阈值得到的TPR有负相关")
        print(np.corrcoef(class_tpr_dict.get(aug_type), class_tpr_global_dict.get(aug_type)))



        for class_label in classes:
            class_indices = np.where(true_labels_per_class == class_label)[0]
            class_phy = phy[class_indices]
            class_indicator = indicator[class_indices]
            #target_class_indicator = indicator_target[class_indices]
            #target_class_phy = target_phy[class_indices]


            fpr, tpr, thresholds = roc_curve(class_indicator, class_phy)

            for i in range(len(fpr) - 1, -1, -1):
                if fpr[i] <= 1e-3:
                    tpr_0_1_class = tpr[i] * 100
                    break
            for i in range(len(fpr) - 1, -1, -1):
                if fpr[i] <= 1e-4:
                    tpr_00_1_class = tpr[i] * 100
                    break
            for i in range(len(fpr) - 1, -1, -1):
                if fpr[i] <= 1e-5:
                    tpr_000_1_class = tpr[i] * 100
                    break

            class_tprat01fpr_global_dict[aug_type].append(tpr_0_1_class)
            class_tprat0001fpr_global_dict[aug_type].append(tpr_000_1_class)
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
    for i, ((aug, accs), (aug_g, accs_same),(aug_g, accs_same_tpr01fpr),(aug_g, accs_same_tpr0001fpr)) in enumerate(zip(class_tpr_dict.items(), class_tpr_global_dict.items(),class_tprat01fpr_global_dict.items(),class_tprat0001fpr_global_dict.items())):
    #for i, ((aug, accs), (aug_g, accs_same)) in enumerate(zip(class_accs_dict.items(), class_accs_global_dict.items())):
        color = colors[i % len(colors)]  # 循环使用颜色列表中的颜色
        #plt.plot(classes, accs, marker='o', label=aug + '-class-opt-thr', color=color)
        #plt.plot(classes, accs_same, marker='^', linestyle='--', label=aug_g + '-glo-opt-thr', color=color)
        #plt.plot(classes, accs, linestyle='--',label=aug + '-class-opt-thr', color=color)
        #plt.plot(classes, accs_same, label=aug_g + '-global-opt-thr', color=color)
        #plt.axhline(y=aug_optimal_mia_acc[i], linestyle=':', label=aug + '-overall', color=colors[i])
        if not os.path.exists(f"acc/mia/lira/TPR/%s/%s" % (DATASET,aug_g)):
            os.makedirs(f"acc/mia/lira/TPR/%s/%s" % (DATASET,aug_g))

        if CLASS_WISE:
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_class.npy" % (DATASET,aug_g,aug_g,MODELNUMBER), accs_same)
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_tpr01fpr_class.npy" % (DATASET,aug_g,aug_g,MODELNUMBER), accs_same_tpr01fpr)
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_tpr0001fpr_class.npy" % (DATASET,aug_g, aug_g, MODELNUMBER), accs_same_tpr0001fpr)
        else:
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_mem.npy" % (DATASET,aug_g, aug_g, MODELNUMBER), accs_same)
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_tpr01fpr_mem.npy" % (DATASET,aug_g, aug_g, MODELNUMBER), accs_same_tpr01fpr)
            np.save("acc/mia/lira/TPR/%s/%s/lira_%s_%s_tpr0001fpr_mem.npy" % (DATASET,aug_g, aug_g, MODELNUMBER), accs_same_tpr0001fpr)

    # 设置图的标题和标签
    #plt.title('TPR rate of %s MIA' % (atktype))
    #plt.xlabel('memory value')
    #plt.ylabel('TPR')
    #plt.ylim(0.54, 0.7)
    #plt.xticks(classes, range(len(classes)))  # 设置 x 轴刻度
    #plt.legend()  # 显示图例
    #plt.grid(True)  # 显示网格线
    #plt.savefig('Figure/' + atktype + '.png')

    return Acc_opt_glo,tpr_glo_opt,tpr_0_1_class_glo_opt,tpr_00_1_class_glo_opt,tpr_000_1_class_glo_opt,tpr_0_0_class_glo_opt,tpr_glo,fpr_glo

    #plt.show()


def privacy_leak_each_class_model_mixup_lam():
    aug_types = ['mixup']
    #atk_lams =["50","51","55","60","65","70","75","80","85","90","95","99"]
    atk_lams = ["51"]
    thr_type_opt = {'LiRA':-0.108,'mixup':-32.316,'pgdat': 1.146, 'base': 3.262, 'AWP': 0.069, 'trades': -0.300,'TradesAWP': -0.850}
    #phy_diff : 5.117 64%  loss:5.746 62% 64% 7.553

    #thr_type_opt = {'pgdat':3.570,'base':5.161}
    #thr_type_opt = {'pgdat':1.146,'base':0.794}
    #loss
    #thr_type_opt = {'pgdat': 9.08, 'base': 9.851, 'AWP': 7.975, 'trades': 5.801,'TradesAWP': 5.169}
    dataset = 'cifar100'
    atktype = ('mixupmem0_phy_diff')


    class_accs_dict = {aug_type: [] for aug_type in atk_lams}
    class_accs_global_dict = {aug_type: [] for aug_type in atk_lams}
    class_tpr_dict = {aug_type: [] for aug_type in atk_lams}
    class_tpr_global_dict = {aug_type: [] for aug_type in atk_lams}
    agunum = 0
    aug_optimal_mia_acc = []

    for aug_type in atk_lams:
        print("Augmentation type: {}".format("mixup"))
        trial = 0

        dirs = os.path.join(root, training_model)
        print(dirs,atktype,MODELNUMBER)
        phy = np.load("%s/%s_%s_%s.npy" % (dirs,atktype,MODELNUMBER,aug_type))


        #phy = np.load("sampleinfo/128model_baseeval_target0.npy")
        #target_phy = np.load("%s/%s_1.npy" % (dirs,atktype))

        # 加载类别标签
        true_labels_per_class = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")
        #true_labels_per_class = np.load("sampleinfo/cifar100_train_labels.npy")

        # 假设 phy、indicator、sample_num 和 samplelist 已定义
        indicator = np.zeros(sample_num, dtype=np.bool_)
        #indicator_target = np.zeros(sample_num, dtype=np.bool_)
        indicator[samplelist_target] = True
        y = np.zeros(sample_num)
        y[samplelist_target] = int(1)
        #indicator[samplelist[int(MODELNUMBER)]] = True

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

        ROC_curve(y, phy,training_model,aug_type,atktype)

        # 计算整体的 ROC AUC、准确率和最佳阈值
        fpr, tpr, thresholds = roc_curve(indicator, phy)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr,roc_auc,atktype,training_model,aug_type)
        AccList = 1 - np.logical_xor(phy[:, np.newaxis] > thresholds[np.newaxis, :], indicator[:, np.newaxis]).sum(0) / len(phy)
        Acc_opt = np.max(AccList)
        ind = np.argmax(AccList)
        thr_opt = thresholds[ind]
        #AccList_target = 1 - np.logical_xor(target_phy[:, np.newaxis] > thr_opt, indicator_target[:, np.newaxis]).sum(0) / len(target_phy)
        # 输出整体的 ROC AUC、准确率和最佳阈值
        print("Overall: Acc = {:.3f}, Threshold = {:.3f}".format(Acc_opt, thr_opt))
        aug_optimal_mia_acc.append(Acc_opt)

        #print("Global opt的阈值和选用Class opt的阈值得到的TPR有负相关")
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
            thr_cls_opt = thresholds[ind]
            tpr_opt = tpr[ind]
            class_tpr[class_label] = tpr_opt
            class_thresholds[class_label] = thr_cls_opt

            _, tpr_cls, _ = roc_curve(class_indicator, phy[class_indices] > thr_cls_opt)

            #target_accList = 1 - np.logical_xor(target_class_phy[:, np.newaxis] > thr_opt,target_class_indicator[:, np.newaxis]).sum(0) / len(target_class_phy)
            class_accs[class_label] = Acc_opt
            class_tpr_dict[aug_type].append(tpr_cls[1])
            class_accs_dict[aug_type].append(Acc_opt)


            # 使用全局最佳阈值计算 ROC 曲线
            #print(thr_opt)
            fpr_global, tpr_global, _ = roc_curve(class_indicator, phy[class_indices] > thr_opt)
            roc_auc_global = auc(fpr_global, tpr_global)
            class_roc_curves_global[class_label] = (fpr_global, tpr_global)
            class_roc_aucs_global[class_label] = roc_auc_global
            AccList_g = 1 - np.logical_xor(class_phy[:, np.newaxis] > thr_opt, class_indicator[:, np.newaxis]).sum(0) / len(class_phy)
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
        #plt.plot(classes, accs_same, label=aug_g + '-global-opt-thr', color=color)
        #plt.plot(classes, accs_same, label='lam 0.' + aug_g, color=color)
        #plt.axhline(y=aug_optimal_mia_acc[i], linestyle=':', label=aug + '-overall', color=colors[i])




    #np.save('privacyacc/cifar100/mia/miatpr_base_clsthr.npy', class_tpr_dict.get('pgdat'))
    #np.save('privacyacc/cifar100/mia/miatpr_base_glothr.npy', class_tpr_global_dict.get('pgdat'))

    #plt.plot(class_labels, pgd_org, marker='^', linestyle='--', color='tab:blue', label='pgdat-test')
    #plt.plot(class_labels, org_org, marker='^', linestyle='--', color='tab:orange', label='base-test')

    # 设置图的标题和标签
    #plt.title('TPR rate of %s MIA' % (atktype))
    #plt.xlabel('memory value')
    #plt.ylabel('TPR')
    #plt.ylim(0.54, 0.7)
    #plt.xticks(classes, range(len(classes)))  # 设置 x 轴刻度
    #plt.legend()  # 显示图例
    #plt.grid(True)  # 显示网格线
    #plt.savefig('Figure/' + training_model + '/' +atktype + '.png')
    #plt.show()


def cal_ic(data):
    std_dev = np.std(data, ddof=1)

    standard_error = std_dev / np.sqrt(len(data))
    return 1.96 * standard_error

BEST_MODEL= False
CLASS_WISE = False
AUG_TYPES = ["trades_reg_8_0.2"]
if __name__ == '__main__':

    #compute_feature("pgdat")
    #
    all_tpr = []
    all_fpr = []
    all_tpr2 = []
    all_fpr2 = []
    overalltpr = []
    overalltpr01fpr = []
    overalltpr001fpr = []
    overalltpr0001fpr = []
    overalltpr0fpr = []
    balance_acc = []
    random_model = [str(random.randint(0, 127)) for _ in range(10)]


    for num in random_model:
        print(num)
        print(num)
        print(num)
        print(num)
        acc,tpr,tpr01fpr,tpr001fpr,tpr0001fpr,tpr0fpr,tpr_glo,fpr_glo = privacy_leak_each_class_model(num,CLASS_WISE,aug_types = AUG_TYPES)
        balance_acc.append(acc)
        overalltpr.append(tpr)
        overalltpr01fpr.append(tpr01fpr)
        overalltpr001fpr.append(tpr001fpr)
        overalltpr0001fpr.append(tpr0001fpr)
        overalltpr0fpr.append(tpr0fpr)
        all_tpr.append(tpr_glo)
        all_fpr.append(fpr_glo)

    #for num in MODELNUMBER_soli8:
        #_,_,_,_,_,_,tpr_glo2,fpr_glo2 = privacy_leak_each_class_model(num,CLASS_WISE,aug_types = ["pgdat_8"])

        #all_tpr2.append(tpr_glo2)
        #all_fpr2.append(fpr_glo2)

    #privacy_leak_each_class_model_mixup_lam()
    #privacy_leak_each_class_entr_target_model()
    #writeit()
    #plot_acc_bias()\
    #np.save(f"scala_figure/%s_tpr01fpr.npy"%(AUG_TYPES[0]),overalltpr01fpr)


    if overalltpr and overalltpr01fpr:  # 确保列表不为空
        print(
            f"balance acc: %.4f final tpr: %.4f, final tpr @0.1 fpr: %.4f,final tpr @0.01 fpr: %.4f,final tpr @0.001 fpr: %.4f, tpr @0.0 fpr: %.4f" % (np.mean(balance_acc)*100, np.mean(overalltpr)*100, np.mean(overalltpr01fpr),np.mean(overalltpr001fpr),np.mean(overalltpr0001fpr),np.mean(overalltpr0fpr)))

        print(
            f"balance acc: %.4f final tpr: %.4f, final tpr @0.1 fpr: %.4f,final tpr @0.01 fpr: %.4f,final tpr @0.001 fpr: %.4f, tpr @0.0 fpr: %.4f" % (
            cal_ic(balance_acc)* 100, cal_ic(overalltpr) * 100 , cal_ic(overalltpr01fpr), cal_ic(overalltpr001fpr),
            cal_ic(overalltpr0001fpr), cal_ic(overalltpr0fpr)))

    # 创建一个新的图表
    #plt.figure(figsize=(8, 6))

    # 绘制所有的ROC曲线
    #for i in range(10):
        #plt.plot(all_fpr[i], all_tpr[i], color="tab:blue",lw=2, label=f'ROC curve {i + 1}')

        #plt.plot(all_fpr2[i], all_tpr2[i], color="tab:orange", lw=2, label=f'ROC curve {i + 1}')

    # 添加对角线（随机猜测）
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim([1e-4, 1])
    #plt.ylim([0.4, 1])
    # 添加标题和标签
    #plt.title('10 ROC Curves')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')

    # 显示图例
    #plt.legend(loc='lower right')

    # 显示网格
    #plt.grid(True)

    # 保存图表为SVG格式
    #plt.savefig("10_roc_curves.svg", format='svg')

    # 显示图表
    #plt.show()