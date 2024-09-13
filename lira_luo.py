# from others_project.lixiao.privacy_and_aug.models import ResNet18
import os
import random
import numpy as np

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']  # 定义颜色列表
root = "/data/luo/reproduce/privacy_and_aug"
sample_num = 60000
class_labels = list(range(100))


DATASET = "svhn"
with open("sampleinfo/samplelist.txt", "r") as f:
    samplelist = eval(f.read())


def normal(mu, sigma2, x):
    r = - np.log(sigma2) / 2 - (x - mu) ** 2 / (2 * sigma2)
    return r


def compute_feature_original(aug_type,trial,test_ids=0):
    if BEST_MODEL:
        dirs = os.path.join(root, "phy_best", DATASET, aug_type)
    else:
        dirs = os.path.join(root, "phy", DATASET, aug_type)

    allmodellist = list(range(0, 128))
    allmodellist.remove(int(trial))

    IN = []
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
        npdict[index] = np.load("%s/phy_%s.npy" % (dirs, str(index)))
        #[:, test_ids]
        if len(npdict[index]) != 60000:
            print("wrong!!!!")
            print(index)
        #print("%s/phy_multi_%s.npy" % (dirs, str(index)))

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

        confsin = confsin[~np.isnan(confsin)]
        confsout = confsout[~np.isnan(confsout)]
        in_u, in_sigma = np.mean(confsin), np.var(confsin)
        out_u, out_sigma = np.mean(confsout), np.var(confsout)
        in_dict[i] = (in_u, in_sigma)
        out_dict[i] = (out_u, out_sigma)
        confs_dict[i] = (confsin, confsout)

    print('test one model')
    base = np.load("%s/phy_%s.npy" % (dirs, trial))
    #[:, test_ids]
    base_in = []
    base_out = []
    for i in range(sample_num):
        a = normal(in_dict[i][0], in_dict[i][1], base[i])
        b = normal(out_dict[i][0], out_dict[i][1], base[i])
        base_in.append(a)
        base_out.append(b)
    base_in = np.array(base_in)
    base_out = np.array(base_out)
    baseeval = base_in - base_out
    return base, baseeval, confs_dict


BEST_MODEL = False

if __name__ == '__main__':
    aug_type = "trades_reg_8_0.2"
    random_model = [str(random.randint(0, 127)) for _ in range(10)]
    for number in random_model:
        base,save_pth,conf = compute_feature_original(aug_type,number)
        if not os.path.exists(f"acc/mia/%s/%s" %(DATASET,aug_type)):
            os.makedirs(f"acc/mia/%s/%s" %(DATASET,aug_type))
        if BEST_MODEL:
            np.save(f"acc/mia/%s/%s/lira_num%s_%s_best_phy.npy" % (DATASET,aug_type, number, aug_type), save_pth)
        else:
            np.save(f"acc/mia/%s/%s/lira_num%s_%s_phy.npy" %(DATASET,aug_type,number,aug_type),save_pth)