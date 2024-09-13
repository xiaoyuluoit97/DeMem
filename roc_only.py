import os
import numpy as np
import torch
from dataset import get_cifar100_datasets
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc, roc_auc_score



def ROC_curve(sample_inds, pred, model,save=False, show=True, name="test", aug_type=None):
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
        ax1.set_title('AUROC %.2f tpr@0.1fpr: %.2f' % (auroc_log, tpr_0_1))
        ax1.plot([1e-7, 1], [1e-7, 1], 'r--')
        ax1.set_xlim([1e-7, 1])
        ax1.set_ylim([1e-7, 1])
        ax1.set_ylabel('True Positive Rate')
        ax1.set_xlabel('False Positive Rate')

        # Linear scale plot
        ax2.plot(fpr, tpr, 'k')
        ax2.set_title('AUROC %.2f  tpr@0.1fpr: %.2f' % (auroc, tpr_0_1))
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('True Positive Rate')
        ax2.set_xlabel('False Positive Rate')

        plt.tight_layout()

        if save:
            plt.savefig(name + ".pdf", dpi=300)
        else:
            plt.savefig('Figure/' + model + '/' + '.png')
            plt.show()



if __name__ == "__main__":
    feature_score = np.load("/Users/xiaoyuluo/workspace/privacy_and_aug-main/mia/rmia_conf/77_rmia_score.npy")
    indicator = np.load("/Users/xiaoyuluo/workspace/privacy_and_aug-main/mia/rmia_conf/target_model_77_indicator_rmia_score.npy")

    ROC_curve(indicator,-feature_score,"resnet18",)