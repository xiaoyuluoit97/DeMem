from audioop import mul
import numpy as np
import os, json, time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import datetime
import matplotlib.pyplot as plt
import utils
from models import ResNet18, MLP, CNN
from models.AlexNet import AlexNet
from dataset import get_loaders, root, get_shuffle_loaders,get_mixupmem_loaders,get_squ_loaders,get_mixupmem_loaders_hu
from advtrain import cal_adv
from tqdm import tqdm


LAM = "mixupfilter"
LAYER = 1
feature_mixup = False
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training models.")
    parser.add_argument('--exp_name', type=str, default=root)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--load_best_model', action='store_true', default=False)
    parser.add_argument('--data_parallel', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--dataset', default='cifar100',
                        choices=["cifar10", "cifar100", "svhn", "purchase", "locations"])

    parser.add_argument('--cuda', default=0, type=int, help='perturbation bound')
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation bound')
    parser.add_argument('--s_model', default=0, type=int, help='the index of the first model')
    parser.add_argument('--t_model', default=1, type=int, help='the index of the last model')
    parser.add_argument('--num_model', default=0, type=str, help='number of target model')
    parser.add_argument('--aug_type', default="", type=str, help='aug type')
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--mode', default="train", choices=["all", "train", "target", "eval"])

    parser.add_argument('--training_model', default="resnet18", choices=["resnet18", "alexnet"])
    parser.add_argument('--aug_method', default="train", choices=["crop", "mixup", "conf", "jitter","mixupmem0","mixupmem1","mixupthree","mixuphu"])

    parser.add_argument('--query_mode', default="single", choices=["single", "multiple", "white"])

    parser.add_argument('--without_base', action='store_true', default=False)
    parser.add_argument('--cnn', action='store_true', default=False)
    return parser.parse_args()


args = get_arguments()

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda}')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')



def SoftLabelNLL(predicted, target, reduce=False):
    if reduce:
        return -(target * predicted).sum().mean()
    else:
        return -(target * predicted).sum()


def _log_value(probs, small_value=1e-30):
    return -np.log(np.maximum(probs.detach().cpu().numpy(), small_value))


def _entr_comp(probs):
    log_probs = _log_value(probs)
    return torch.sum(np.multiply(probs, log_probs))

def _m_entr_comp(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1 - probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[true_labels] = reverse_probs[true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[true_labels] = log_probs[true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs))

def visual(picnum, b_x, imgs, imgs2, cids, cids2, b_cid):
    data = np.load('sampleinfo/cifar100_infl_matrix.npz')
    traing_index_classidx_1 = data['tr_classidx_1']
    testing_index_classidx_1 = data['tt_classidx_1']
    tr_labels = data['tr_labels']
    tr_mem = data['tr_mem']

    b_x = b_x[:picnum].cpu().numpy()
    imgs = imgs[:picnum].cpu().numpy()
    #imgs2 = imgs2[:picnum].cpu().numpy()

    cids = tr_labels[:picnum]
    mem = tr_mem[:picnum]
    cids2 = cids2[:picnum].cpu().numpy()
    b_cids = b_cid[:picnum].cpu().numpy()

    fig, axes = plt.subplots(3, picnum, figsize=(20, 6))

    for i in range(picnum):
        # 显示原始图像
        img = imgs[i].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[0, i].imshow(img)  # 修改这里的索引为 i
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Ori: {cids[i]}\n{mem[i]:.2f}')

        #img2 = imgs2[i].transpose(1, 2, 0)
        #img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        #axes[1, i].imshow(img2)  # 修改这里的索引为 i
        #axes[1, i].axis('off')
        #axes[1, i].set_title(f'Ori2: {cids2[i]}')

        img_aug = b_x[i].transpose(1, 2, 0)
        img_aug = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())
        axes[2, i].imshow(img_aug)  # 修改这里的索引为 i
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Aug: {b_cids[i]}')

    plt.show()
    print("showshowshow")

def feature_visual(picnum, b_x, imgs, imgs2, cids, cids2, b_cid, model):
    feature_maps_bx = model.get_feature_maps(b_x)
    feature_maps_img1 = model.get_feature_maps(imgs)
    feature_maps_img2 = model.get_feature_maps(imgs2)

    processed_feature_maps = {
        "bx": [],
        "img1": [],
        "img2": []
    }
    layer_names = list(feature_maps_bx.keys())  # 提取特征图层的名称

    # 处理特征图
    for name in layer_names:
        feature_map_bx = feature_maps_bx[name][:1].squeeze(0)
        feature_map_img1 = feature_maps_img1[name][:1].squeeze(0)
        feature_map_img2 = feature_maps_img2[name][:1].squeeze(0)

        mean_feature_map_bx = torch.mean(feature_map_bx, dim=0)  # 计算通道的平均值
        mean_feature_map_img1 = torch.mean(feature_map_img1, dim=0)
        mean_feature_map_img2 = torch.mean(feature_map_img2, dim=0)

        processed_feature_maps["bx"].append(mean_feature_map_bx.data.cpu().numpy())
        processed_feature_maps["img1"].append(mean_feature_map_img1.data.cpu().numpy())
        processed_feature_maps["img2"].append(mean_feature_map_img2.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 20))
    num_layers = len(layer_names)

    # 显示原始图像
    bx = b_x[:1].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
    bx = (bx - bx.min()) / (bx.max() - bx.min())  # 归一化图像到 [0, 1] 范围
    ax = fig.add_subplot(3, num_layers + 1, 1)
    ax.imshow(bx)
    ax.axis("off")

    img = imgs[:1].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
    img = (img - img.min()) / (img.max() - img.min())  # 归一化图像到 [0, 1] 范围
    ax = fig.add_subplot(3, num_layers + 1, num_layers + 2)
    ax.imshow(img)
    ax.axis("off")

    img2 = imgs2[:1].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())  # 归一化图像到 [0, 1] 范围
    ax = fig.add_subplot(3, num_layers + 1, 2 * (num_layers + 1)+1)
    ax.imshow(img2)
    ax.axis("off")

    # 显示特征图
    for i in range(num_layers):
        ax = fig.add_subplot(3, num_layers + 1, i + 2)
        ax.imshow(processed_feature_maps["bx"][i], cmap='viridis')
        ax.axis("off")
        ax.set_title(f"{layer_names[i]} - bx", fontsize=15)

        ax = fig.add_subplot(3, num_layers + 1, num_layers + i + 3)
        ax.imshow(processed_feature_maps["img1"][i], cmap='viridis')
        ax.axis("off")
        ax.set_title(f"{layer_names[i]} - img1", fontsize=15)

        ax = fig.add_subplot(3, num_layers + 1, 2 * (num_layers + 1) + i + 2)
        ax.imshow(processed_feature_maps["img2"][i], cmap='viridis')
        ax.axis("off")
        ax.set_title(f"{layer_names[i]} - img2", fontsize=15)

    plt.tight_layout()
    plt.show()


    data = np.load('sampleinfo/cifar100_infl_matrix.npz')
    traing_index_classidx_1 = data['tr_classidx_1']
    testing_index_classidx_1 = data['tt_classidx_1']
    # each datapoints label
    # each test point label
    tr_labels = data['tr_labels']
    tr_mem = data['tr_mem']




    b_x = b_x[:picnum].cpu().numpy()
    imgs = imgs[:picnum].cpu().numpy()
    imgs2 = imgs2[:picnum].cpu().numpy()# 原始图像


    cids = tr_labels[:picnum]  # 原始图像的类ID
    mem = tr_mem[:picnum]
    cids2 = cids2[:picnum].cpu().numpy()
    b_cids = b_cid[:picnum].cpu().numpy()  # 增强后图像的类ID

    # 可视化前五个经过数据增强的图像及其原始图像
    fig, axes = plt.subplots(3, picnum, figsize=(20, 6))
    for i in range(picnum):
        # 显示原始图像
        img = imgs[i].transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化图像到 [0, 1] 范围
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Ori: {cids[i]}\n{mem[i]:.2f}')

        img2 = imgs2[i].transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())  # 归一化图像到 [0, 1] 范围
        axes[1, i].imshow(img2)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Ori2: {cids2[i]}')

        # 显示增强后的图像
        img_aug = b_x[i].transpose(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img_aug = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())  # 归一化图像到 [0, 1] 范围
        axes[2, i].imshow(img_aug)
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Aug: {b_cids[i]}')

    plt.show()


def test_three_mixup(epoch, model, testloader, testshuffleloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    to_save = []
    loss_to_save = []
    lossclean_list = []
    entr_to_save = []
    conf_to_save = []
    m_entr_to_save = []
    iterator = zip(testloader, testshuffleloader,testshuffleloader)

    for batch in iterator:
        #imgs, cids = batch
        #imgs = imgs.clone()
        #imgs, cids = imgs.to(device), cids.to(device)
        if args.query_mode == "multiple":
            imgs_2 = imgs.clone().flip(4)
            imgs = torch.cat([imgs, imgs_2], dim=1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.aug_type == "base":
            (imgs, cids), (imgs_2, cids_2),(imgs_3, cids_3) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)
            imgs_3, cids_3 = imgs_3.to(device), cids_3.to(device)

            lam = LAM


        if args.save_results:
            with torch.no_grad():
                b_x = (lam * imgs) + ((1 - lam) / 2 * imgs_2) + ((1 - lam) / 2 * imgs_3)
                pred_aug = model.base_forward(b_x)
                pred_clean = model.base_forward(imgs)


            logits_clean = F.softmax(pred_clean, dim=1)
            logits_aug = F.softmax(pred_aug, dim=1)

            iss = torch.arange(pred_clean.shape[0])

            losses_clean = torch.zeros(pred_clean.size(0))
            losses_aug = torch.zeros(pred_aug.size(0))

            entr_clean = torch.zeros(pred_clean.size(0))
            entr_aug = torch.zeros(pred_aug.size(0))

            m_entres_clean = torch.zeros(pred_clean.size(0))
            m_entres_aug = torch.zeros(pred_aug.size(0))

            entr_logits_clean = logits_clean.cpu()
            entr_logits_aug = logits_aug.cpu()


            for i in range(pred_clean.size(0)):
                loss_clean = -criterion(pred_clean[i], cids[i])
                loss_aug = -criterion(pred_aug[i], cids[i])

                losses_clean[i] = loss_clean.item()
                losses_aug[i] = loss_aug.item()


                m_entres_clean[i] = _m_entr_comp(entr_logits_clean[i], cids[i]).item()
                m_entres_aug[i] = _m_entr_comp(entr_logits_aug[i], cids[i]).item()

                entr_clean[i] = _entr_comp(entr_logits_clean[i]).item()
                entr_aug[i] = _entr_comp(entr_logits_aug[i]).item()


            phy_clean = torch.log(logits_clean[iss, cids[iss]])

            conf = logits_clean[iss, cids[iss]]

            phy_aug = torch.log(logits_aug[iss, cids[iss]])

            pred_clean = F.log_softmax(logits_clean, dim=1)
            pred_aug = F.log_softmax(logits_aug, dim=1)


            logits_clean[iss, cids[iss]] = 0
            logits_aug[iss, cids[iss]] = 0

            phy_clean = phy_clean - torch.log(torch.sum(logits_clean, dim=1) + 1e-20)
            phy_aug = phy_aug - torch.log(torch.sum(logits_aug, dim=1) + 1e-20)

            phy_diff = phy_clean - phy_aug
            phy_diff = phy_diff.cpu().numpy()

            losses_diff =  losses_clean - losses_aug

            losses_diff = losses_diff.cpu().numpy()
            losses_clean = losses_clean.cpu().numpy()
            m_entres_diff = m_entres_clean - m_entres_aug

            m_entres_diff = m_entres_diff.cpu().numpy()

            entr_diff = entr_clean - entr_aug
            entr_diff = entr_diff.cpu().numpy()

            conf = conf.cpu().numpy()

            conf_to_save.append(conf)
            entr_to_save.append(entr_diff)
            m_entr_to_save.append(m_entres_diff)
            loss_to_save.append(losses_diff)
            lossclean_list.append(losses_clean)
            to_save.append(phy_diff)


        else:
            with torch.no_grad():
                pred_clean = model(imgs)

        loss_clean = criterion(pred_clean, cids)

        _, predicted = pred_clean.max(1)
        category_total += cids.size(0)
        category_loss += loss_clean.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
    if args.save_results:
        conflist = np.concatenate(conf_to_save)
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        lossclean_list = np.concatenate(lossclean_list)
        entrlist = np.concatenate(entr_to_save)
        m_entrlist = np.concatenate(m_entr_to_save)

        return 100. * (category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist,m_entrlist,conflist,lossclean_list

    return 100. * (category_correct / category_total), category_loss / category_total


def test_feature_mixup(epoch, model, testloader, testshuffleloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    lossclean_list = []
    entr_to_save = []
    conf_to_save = []
    m_entr_to_save = []
    iterator = zip(testloader, testshuffleloader)

    for batch in iterator:
        #imgs, cids = batch
        #imgs = imgs.clone()
        #imgs, cids = imgs.to(device), cids.to(device)
        if args.query_mode == "multiple":
            imgs_2 = imgs.clone().flip(4)
            imgs = torch.cat([imgs, imgs_2], dim=1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.aug_type == "base":
            (imgs, cids), (imgs_2, cids_2) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)

            alpha = cfg['augmentation_params']['mixup'][0]
            lam = np.random.beta(5, 0.5)
            lam = LAM
            b_x = (lam * imgs) + ((1 - lam) * imgs_2)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                   cids.view(
                                                                                                                       -1,
                                                                                                                       1),
                                                                                                                   1)
            b_y_one_hot_2 = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                     cids_2.view(
                                                                                                                         -1,
                                                                                                                         1),
                                                                                                                     1)


            b_cid = (lam * b_y_one_hot) + ((1 - lam) * b_y_one_hot_2)
            b_cid = torch.argmax(b_cid, axis=1)
            #visual(20,b_x,imgs,imgs_2,cids,cids_2,b_cid)
            #pred = model(b_x)
            #loss = criterion(pred, b_cid)

        if args.save_results:
            with torch.no_grad():
                if feature_mixup:
                    imgs_layer1out = model.base_forward_feature(imgs  , LAYER)
                    imgs2_layer1out = model.base_forward_feature(imgs_2, LAYER)

                    b_x_layer1out = (lam * imgs_layer1out) + ((1 - lam) * imgs2_layer1out)


                    pred_aug = model.continue_forward_feature(b_x_layer1out,LAYER+1)
                    pred_clean = model.continue_forward_feature(imgs_layer1out,LAYER+1)
                else:
                    b_x = (lam * imgs) + ((1 - lam) * imgs_2)
                    pred_aug = model.base_forward(b_x)
                    pred_clean = model.base_forward(imgs)


            logits_clean = F.softmax(pred_clean, dim=1)
            logits_aug = F.softmax(pred_aug, dim=1)

            iss = torch.arange(pred_clean.shape[0])

            losses_clean = torch.zeros(pred_clean.size(0))
            losses_aug = torch.zeros(pred_aug.size(0))

            entr_clean = torch.zeros(pred_clean.size(0))
            entr_aug = torch.zeros(pred_aug.size(0))

            m_entres_clean = torch.zeros(pred_clean.size(0))
            m_entres_aug = torch.zeros(pred_aug.size(0))

            entr_logits_clean = logits_clean.cpu()
            entr_logits_aug = logits_aug.cpu()


            for i in range(pred_clean.size(0)):
                loss_clean = -criterion(pred_clean[i], cids[i])
                loss_aug = -criterion(pred_aug[i], b_cid[i])

                losses_clean[i] = loss_clean.item()
                losses_aug[i] = loss_aug.item()


                m_entres_clean[i] = _m_entr_comp(entr_logits_clean[i], cids[i]).item()
                m_entres_aug[i] = _m_entr_comp(entr_logits_aug[i], b_cid[i]).item()

                entr_clean[i] = _entr_comp(entr_logits_clean[i]).item()
                entr_aug[i] = _entr_comp(entr_logits_aug[i]).item()


            phy_clean = torch.log(logits_clean[iss, cids[iss]])

            conf = logits_clean[iss, cids[iss]]

            phy_aug = torch.log(logits_aug[iss, b_cid[iss]])

            pred_clean = F.log_softmax(logits_clean, dim=1)
            pred_aug = F.log_softmax(logits_aug, dim=1)


            logits_clean[iss, cids[iss]] = 0
            logits_aug[iss, cids[iss]] = 0

            phy_clean = phy_clean - torch.log(torch.sum(logits_clean, dim=1) + 1e-20)
            phy_aug = phy_aug - torch.log(torch.sum(logits_aug, dim=1) + 1e-20)

            phy_diff = phy_clean - phy_aug
            phy_diff = phy_diff.cpu().numpy()

            losses_diff =  losses_clean - losses_aug

            losses_diff = losses_diff.cpu().numpy()
            losses_clean = losses_clean.cpu().numpy()
            m_entres_diff = m_entres_clean - m_entres_aug

            m_entres_diff = m_entres_diff.cpu().numpy()

            entr_diff = entr_clean - entr_aug
            entr_diff = entr_diff.cpu().numpy()

            conf = conf.cpu().numpy()

            conf_to_save.append(conf)
            entr_to_save.append(entr_diff)
            m_entr_to_save.append(m_entres_diff)
            loss_to_save.append(losses_diff)
            lossclean_list.append(losses_clean)
            to_save.append(phy_diff)



        else:
            with torch.no_grad():
                pred_clean = model(imgs)

        loss_clean = criterion(pred_clean, cids)

        _, predicted = pred_clean.max(1)
        category_total += cids.size(0)
        category_loss += loss_clean.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
    if args.save_results:
        conflist = np.concatenate(conf_to_save)
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        lossclean_list = np.concatenate(lossclean_list)
        entrlist = np.concatenate(entr_to_save)
        m_entrlist = np.concatenate(m_entr_to_save)

        return 100. * (category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist,m_entrlist,conflist,lossclean_list

    return 100. * (category_correct / category_total), category_loss / category_total


def test_mixup(epoch, model, testloader, testshuffleloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    lossclean_list = []
    entr_to_save = []
    conf_to_save = []
    m_entr_to_save = []
    iterator = zip(testloader, testshuffleloader)

    for batch in iterator:
        #imgs, cids = batch
        #imgs = imgs.clone()
        #imgs, cids = imgs.to(device), cids.to(device)
        if args.query_mode == "multiple":
            imgs_2 = imgs.clone().flip(4)
            imgs = torch.cat([imgs, imgs_2], dim=1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.aug_type == "base":
            (imgs, cids), (imgs_2, cids_2) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)

            alpha = cfg['augmentation_params']['mixup'][0]
            lam = np.random.beta(5, 0.5)
            lam = LAM
            b_x = (lam * imgs) + ((1 - lam) * imgs_2)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                   cids.view(
                                                                                                                       -1,
                                                                                                                       1),
                                                                                                                   1)
            b_y_one_hot_2 = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                     cids_2.view(
                                                                                                                         -1,
                                                                                                                         1),
                                                                                                                     1)


            b_cid = (lam * b_y_one_hot) + ((1 - lam) * b_y_one_hot_2)
            #b_cid = torch.argmax(b_cid, axis=1)
            #visual(20,b_x,imgs,imgs_2,cids,cids_2,b_cid)
            #pred = model(b_x)
            #loss = criterion(pred, b_cid)

        if args.save_results:
            with torch.no_grad():
                pred_aug = model.base_forward(b_x)
                pred_clean = model.base_forward(imgs)


            logits_clean = F.softmax(pred_clean, dim=1)
            logits_aug = F.softmax(pred_aug, dim=1)

            iss = torch.arange(pred_clean.shape[0])
            #initialize

            losses_clean = torch.zeros(pred_clean.size(0))
            losses_aug = torch.zeros(pred_aug.size(0))

            entr_clean = torch.zeros(pred_clean.size(0))
            entr_aug = torch.zeros(pred_aug.size(0))

            m_entres_clean = torch.zeros(pred_clean.size(0))
            m_entres_aug = torch.zeros(pred_aug.size(0))

            entr_logits_clean = logits_clean.cpu()
            entr_logits_aug = logits_aug.cpu()


            for i in range(pred_clean.size(0)):


                loss_clean = -criterion(pred_clean[i], cids[i])
                loss_aug = -criterion(pred_aug[i], b_cid[i])

                losses_clean[i] = loss_clean.item()
                losses_aug[i] = loss_aug.item()


                m_entres_clean[i] = _m_entr_comp(entr_logits_clean[i], cids[i]).item()
                m_entres_aug[i] = _m_entr_comp(entr_logits_aug[i], b_cid[i]).item()

                entr_clean[i] = _entr_comp(entr_logits_clean[i]).item()
                entr_aug[i] = _entr_comp(entr_logits_aug[i]).item()


            phy_clean = torch.log(logits_clean[iss, cids[iss]])

            conf = logits_clean[iss, cids[iss]]
            #conf = logits_aug[iss, cids[iss]]

            #phy_aug = torch.log(logits_aug[iss, b_cid[iss]])
            phy_aug = torch.log(logits_aug[iss, cids[iss]])
            pred_clean = F.log_softmax(logits_clean, dim=1)
            pred_aug = F.log_softmax(logits_aug, dim=1)


            logits_clean[iss, cids[iss]] = 0
            logits_aug[iss, cids[iss]] = 0

            phy_clean = phy_clean - torch.log(torch.sum(logits_clean, dim=1) + 1e-20)
            phy_aug = phy_aug - torch.log(torch.sum(logits_aug, dim=1) + 1e-20)

            phy_diff = phy_clean - phy_aug
            phy_diff = phy_diff.cpu().numpy()

            losses_diff = losses_aug - losses_clean

            losses_diff = losses_diff.cpu().numpy()
            losses_clean = losses_clean.cpu().numpy()
            #m_entres_diff = m_entres_clean - m_entres_aug

            #m_entres_diff = m_entres_diff.cpu().numpy()

            entr_diff = entr_clean - entr_aug
            entr_diff = entr_diff.cpu().numpy()

            conf = conf.cpu().numpy()

            conf_to_save.append(conf)
            entr_to_save.append(entr_diff)
            #m_entr_to_save.append(m_entres_diff)
            loss_to_save.append(losses_diff)
            lossclean_list.append(losses_clean)
            to_save.append(phy_diff)



        else:
            with torch.no_grad():
                pred_clean = model(imgs)

        #loss_clean = criterion(pred_clean, cids)
        loss_clean = 0.1

        _, predicted = pred_clean.max(1)
        category_total += cids.size(0)
        #category_loss += loss_clean.item()
        category_loss += loss_clean
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        #if (batch_idx + 1) % log_frequency == 0:
            #log_payload = {"category acc": 100. * (category_correct / category_total)}
            #display = utils.log_display(epoch=epoch,
                                        #global_step=ENV["global_step"],
                                        #time_elapse=time_used,
                                        #**log_payload)
            #logger.info(display)
    if args.save_results:
        conflist = np.concatenate(conf_to_save)
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        lossclean_list = np.concatenate(lossclean_list)
        entrlist = np.concatenate(entr_to_save)
        #m_entrlist = np.concatenate(m_entr_to_save)

        return 100. * (category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist,None,conflist,lossclean_list

    return 100. * (category_correct / category_total), category_loss / category_total


def test_mixup_hu(epoch, model, testloader, testaugloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    lossclean_list = []
    entr_to_save = []
    conf_to_save = []
    m_entr_to_save = []
    iterator = zip(testloader, testaugloader)

    for batch in tqdm(iterator):
        #imgs, cids = batch
        #imgs = imgs.clone()
        #imgs, cids = imgs.to(device), cids.to(device)

        start = time.time()

        if args.aug_type == "base":
            (imgs, cids), (imgs_formixup, cids_formixup) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_formixup, cids_formixup = imgs_formixup.to(device), cids_formixup.to(device)

            #visual(20, imgs_aug, imgs, None, cids, cids_aug, cids_aug)

        if args.save_results:
            with torch.no_grad():
                #pred_aug = model.base_forward(imgs_formixup)
                pred_clean = model.base_forward(imgs)


            logits_clean = F.softmax(pred_clean, dim=1)
            #logits_aug = F.softmax(pred_aug, dim=1)
            iss = torch.arange(pred_clean.shape[0])
            #initialize
            losses_clean = torch.zeros(pred_clean.size(0))
            losses_mixedup = torch.zeros(pred_clean.size(0))


            entr_clean = torch.zeros(pred_clean.size(0))
            #entr_aug = torch.zeros(pred_aug.size(0))
            m_entres_clean = torch.zeros(pred_clean.size(0))
            #m_entres_aug = torch.zeros(pred_aug.size(0))
            entr_logits_clean = logits_clean.cpu()
            #entr_logits_aug = logits_aug.cpu()


            for i in range(pred_clean.size(0)):

                index_equal_to_target_label = torch.nonzero(cids_formixup == cids[i], as_tuple=True)[0]
                all_mixup_images = []
                for mixup_index in index_equal_to_target_label:
                    assert cids_formixup[mixup_index] == cids[i]
                    b_x = (0.5 * imgs[i]) + ((1 - 0.5) * imgs_formixup[mixup_index])
                    all_mixup_images.append(b_x)

                all_mixup_images = torch.stack(all_mixup_images)

                with torch.no_grad():
                    pred_aug = model.base_forward(all_mixup_images)
                    logits_aug = F.softmax(pred_aug, dim=1)
                    loss_aug = -criterion(pred_aug, cids[i].repeat(len(pred_aug)))
                    #logits_aug = torch.mean(logits_aug)

                #conf_current = logits_clean[i, cids[i]].item()

                loss_clean = -criterion(pred_clean[i], cids[i])

                losses_clean[i] = loss_clean.item()
                losses_mixedup[i] = loss_aug.item()


            #phy_clean = torch.log(logits_clean[iss, cids[iss]])

            #conf = logits_clean[iss, cids[iss]]
            #conf = logits_aug[iss, cids[iss]]

            #phy_aug = torch.log(logits_aug[iss, b_cid[iss]])
            #phy_aug = torch.log(logits_aug[iss, cids[iss]])
            #pred_clean = F.log_softmax(logits_clean, dim=1)
            #pred_aug = F.log_softmax(logits_aug, dim=1)


            #logits_clean[iss, cids[iss]] = 0
            #logits_aug[iss, cids[iss]] = 0

            #phy_clean = phy_clean - torch.log(torch.sum(logits_clean, dim=1) + 1e-20)
            #phy_aug = phy_aug - torch.log(torch.sum(logits_aug, dim=1) + 1e-20)

            #phy_diff = phy_clean - phy_aug
            #phy_diff = phy_diff.cpu().numpy()

            losses_diff = losses_mixedup - losses_clean

            losses_diff = losses_diff.cpu().numpy()

            losses_clean = losses_clean.cpu().numpy()
            #m_entres_diff = m_entres_clean - m_entres_aug

            #m_entres_diff = m_entres_diff.cpu().numpy()

            #entr_diff = entr_clean - entr_aug
            #entr_diff = entr_diff.cpu().numpy()

            #conf = conf.cpu().numpy()

            #conf_to_save.append(conf)
            #entr_to_save.append(entr_diff)
            #m_entr_to_save.append(m_entres_diff)
            loss_to_save.append(losses_diff)
            lossclean_list.append(losses_clean)
            #to_save.append(phy_diff)


        else:
            with torch.no_grad():
                pred_clean = model(imgs)

        #loss_clean = criterion(pred_clean, cids)
        #loss_clean = 0.1

        _, predicted = pred_clean.max(1)
        category_total += cids.size(0)
        category_loss += loss_clean.item()
        #category_loss += loss_clean
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        #if (batch_idx + 1) % log_frequency == 0:
            #log_payload = {"category acc": 100. * (category_correct / category_total)}
            #display = utils.log_display(epoch=epoch,
                                        #global_step=ENV["global_step"],
                                        #time_elapse=time_used,
                                        #**log_payload)
            #logger.info(display)
    if args.save_results:
        #conflist = np.concatenate(conf_to_save)
        #phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        #lossclean_list = np.concatenate(lossclean_list)
        #entrlist = np.concatenate(entr_to_save)
        phylist = None
        entrlist = None
        #m_entrlist = np.concatenate(m_entr_to_save)

        return 100. * (category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist,None,None,lossclean_list

    return 100. * (category_correct / category_total), category_loss / category_total



def test_augmentation(epoch, model, testloader, testloader_aug, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    entr_to_save = []
    m_entr_to_save = []
    iterator = zip(testloader, testloader_aug)


    for batch in iterator:
        #imgs, cids = batch
        #imgs = imgs.clone()
        #imgs, cids = imgs.to(device), cids.to(device)
        if args.query_mode == "multiple":
            imgs_2 = imgs.clone().flip(4)
            imgs = torch.cat([imgs, imgs_2], dim=1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.aug_type == "base":
            (imgs, cids), (imgs_2, cids_2) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)

            alpha = cfg['augmentation_params']['mixup'][0]
            lam = np.random.beta(5, 0.5)
            lam = LAM
            b_x = (lam * imgs) + ((1 - lam) * imgs_2)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                   cids.view(
                                                                                                                       -1,
                                                                                                                       1),
                                                                                                                   1)
            b_y_one_hot_2 = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device=device).scatter_(1,
                                                                                                                     cids_2.view(
                                                                                                                         -1,
                                                                                                                         1),
                                                                                                                     1)


            b_cid = (lam * b_y_one_hot) + ((1 - lam) * b_y_one_hot_2)
            b_cid = torch.argmax(b_cid, axis=1)
            #visual(20,b_x,imgs,imgs_2,cids,cids_2,b_cid)
            #pred = model(b_x)
            #loss = criterion(pred, b_cid)

        if args.save_results:
            with torch.no_grad():
                pred_aug = model.base_forward(b_x)
                pred_clean = model.base_forward(imgs)

            logits_clean = F.softmax(pred_clean, dim=1)
            logits_aug = F.softmax(pred_aug, dim=1)

            iss = torch.arange(pred_clean.shape[0])

            losses_clean = torch.zeros(pred_clean.size(0))
            losses_aug = torch.zeros(pred_aug.size(0))

            entr_clean = torch.zeros(pred_clean.size(0))
            entr_aug = torch.zeros(pred_aug.size(0))

            m_entres_clean = torch.zeros(pred_clean.size(0))
            m_entres_aug = torch.zeros(pred_aug.size(0))

            entr_logits_clean = logits_clean.cpu()
            entr_logits_aug = logits_aug.cpu()


            for i in range(pred_clean.size(0)):
                loss_clean = -criterion(pred_clean[i], cids[i])
                loss_aug = -criterion(pred_aug[i], b_cid[i])

                losses_clean[i] = loss_clean.item()
                losses_aug[i] = loss_aug.item()


                m_entres_clean[i] = _m_entr_comp(entr_logits_clean[i], cids[i]).item()
                m_entres_aug[i] = _m_entr_comp(entr_logits_aug[i], b_cid[i]).item()

                entr_clean[i] = _entr_comp(entr_logits_clean[i]).item()
                entr_aug[i] = _entr_comp(entr_logits_aug[i]).item()


            phy_clean = torch.log(logits_clean[iss, cids[iss]])
            phy_aug = torch.log(logits_aug[iss, b_cid[iss]])

            pred_clean = F.log_softmax(logits_clean, dim=1)
            pred_aug = F.log_softmax(logits_aug, dim=1)


            logits_clean[iss, cids[iss]] = 0
            logits_aug[iss, cids[iss]] = 0
            phy_clean = phy_clean - torch.log(torch.sum(logits_clean, dim=1) + 1e-20)
            phy_aug = phy_aug - torch.log(torch.sum(logits_aug, dim=1) + 1e-20)

            phy_diff = (phy_clean - phy_aug)/phy_aug
            phy_diff = phy_diff.cpu().numpy()

            losses_diff = losses_aug/losses_clean
            losses_diff = losses_diff.cpu().numpy()

            m_entres_diff = m_entres_clean - m_entres_aug

            m_entres_diff = m_entres_diff.cpu().numpy()

            entr_diff = entr_clean - entr_aug
            entr_diff = entr_diff.cpu().numpy()

            entr_to_save.append(entr_diff)
            m_entr_to_save.append(m_entres_diff)
            loss_to_save.append(losses_diff)
            to_save.append(phy_diff)



        else:
            with torch.no_grad():
                pred_clean = model(imgs)

        loss_clean = criterion(pred_clean, cids)

        _, predicted = pred_clean.max(1)

        category_total += cids.size(0)
        category_loss += loss_clean.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        #if (batch_idx + 1) % log_frequency == 0:
            #log_payload = {"category acc": 100. * (category_correct / category_total)}
            #display = utils.log_display(epoch=epoch,
                                        #global_step=ENV["global_step"],
                                        #time_elapse=time_used,
                                        #**log_payload)
            #logger.info(display)
    if args.save_results:
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        entrlist = np.concatenate(entr_to_save)
        m_entrlist = np.concatenate(m_entr_to_save)

        return 100. * (category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist,m_entrlist

    return 100. * (category_correct / category_total), category_loss / category_total




def main(cfg, aug_type="none", index=0, aug_index=0):
    if args.dataset == "cifar10":
        if args.cnn:
            model = CNN(num_classes=10).to(device)
        else:
            model = ResNet18().to(device)
    elif args.dataset == "cifar100":
        if args.training_model == "resnet18":
            model = ResNet18(num_classes=100).to(device)
        elif args.training_model == "alexnet":
            model = AlexNet(num_classes=100).to(device)
    elif args.dataset == "svhn":
        model = CNN(num_classes=10).to(device)
    elif args.dataset == "purchase":
        model = MLP(num_classes=100, size=600).to(device)
    elif args.dataset == "locations":
        model = MLP(num_classes=30, size=446).to(device)
    assert (aug_type in cfg['training_augmentations'])
    if aug_type in ['distillation', 'smooth', 'mixup']:
        criterion = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    else:
        criterion = nn.NLLLoss()

    test_criterion = nn.NLLLoss()


    if args.exp_name == '':
        new_exp_name = 'exp_' + datetime.datetime.now()
    else:
        if (aug_type == "trades" or aug_type == "pgdat") and args.epsilon < 8:
            new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type + "_" + str(args.epsilon))
        else:
            new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type)

        if args.without_base:
            new_exp_name = new_exp_name + "_none"

    if args.mode == "all" or args.mode == "target":
        index = args.mode

    index = args.num_model
    exp_path = os.path.join(new_exp_name, args.training_model + "_"+ str(index))
    log_file_path = os.path.join(exp_path, args.training_model + "_"+ str(index))

    checkpoint_path = exp_path
    print(exp_path)
    print(checkpoint_path)

    utils.create_path(checkpoint_path)

    logger = utils.setup_logger(name="resnet18_" + str(index), log_file=log_file_path + ".log")
    starting_epoch = 0

    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0}

    if args.load_model:
        checkpoint = utils.load_model(os.path.join(checkpoint_path, 'model'), model)
        starting_epoch = checkpoint[0]['epoch'] + 1
    if args.load_best_model:
        checkpoint = utils.load_model(os.path.join(checkpoint_path, 'model_best'), model)
        starting_epoch = checkpoint[0]['epoch'] + 1
    if aug_type == 'distillation':
        if args.dataset == "cifar10":
            if args.cnn:
                teacher = CNN(num_classes=10).to(device)
            else:
                teacher = ResNet18().to(device)
        elif args.dataset == "cifar100":
            teacher = ResNet18(num_classes=100).to(device)
        elif args.dataset == "svhn":
            teacher = CNN(num_classes=10).to(device)
        elif args.dataset == "purchase":
            teacher = MLP(num_classes=100, size=600).to(device)
        elif args.dataset == "locations":
            teacher = MLP(num_classes=30, size=446).to(device)
        tname = "none" if args.without_base else "base"
        utils.load_model(os.path.join(checkpoint_path.replace("distillation", tname), 'model'), teacher)
        teacher.eval()
    else:
        teacher = None
    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)

    if args.query_mode == "multiple":
        multiple = True
        bs = cfg['regular_batch_size']
    else:
        multiple = False
        bs = cfg['regular_batch_size'] * 4
    if args.aug_method == "mixup":
        testloader, testshuffleloader = get_shuffle_loaders(args.dataset, aug_type, aug_index, cfg,args.training_model,
                                          shuffle=True, batch_size=bs,
                                          multiple=multiple,
                                          without_base=args.without_base)

    elif args.aug_method == "mixupthree":
        testloader, testshuffleloader = get_shuffle_loaders(args.dataset, aug_type, aug_index, cfg,args.training_model,
                                          shuffle=True, batch_size=bs,
                                          multiple=multiple,
                                          without_base=args.without_base)

    elif args.aug_method == "mixupmem0":
        testloader, testshuffleloader = get_mixupmem_loaders(args.dataset, aug_type, aug_index, cfg, args.training_model,
                                                            shuffle=True, batch_size=bs,mem=0,
                                                            multiple=multiple,
                                                            without_base=args.without_base)

    elif args.aug_method == "mixupmem1":
        testloader, testshuffleloader = get_mixupmem_loaders(args.dataset, aug_type, aug_index, cfg,
                                                             args.training_model,
                                                             shuffle=True, batch_size=bs,mem=1,
                                                             multiple=multiple,
                                                             without_base=args.without_base)


    elif args.aug_method == "mixuphu":
        testloader, testaugloader = get_mixupmem_loaders_hu(args.dataset, aug_type, aug_index, cfg, args.training_model,
                                                            shuffle=True, batch_size=bs,mem=0,
                                                            multiple=multiple,
                                                            without_base=args.without_base)

    else:
        testloader, testaugloader = get_squ_loaders(args.dataset, args.aug_method, aug_index, cfg,
                                          shuffle=True, batch_size=bs,
                                          mode=args.mode, samplerindex=index,
                                          multiple=multiple,
                                          without_base=args.without_base)

    logger.info("Starting Epoch: %d" % (starting_epoch))

    if args.save_results:

        if args.aug_method == "mixup":
            if feature_mixup:
                vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist, conflist, lossclean_list = test_feature_mixup(
                    starting_epoch, model, testloader,
                    testshuffleloader, test_criterion,
                    ENV,
                    logger)

            else:
                vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist, conflist, lossclean_list = test_mixup(
                    starting_epoch, model, testloader,
                    testshuffleloader, test_criterion,
                    ENV,
                    logger)
            #vc_acc, vc_loss, phylist, losslist, entrlist,m_entrlist = test_augmentation(starting_epoch, model, testloader, testshuffleloader, test_criterion, ENV,logger)

        elif args.aug_method == "mixuphu":
            vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist, conflist, lossclean_list = test_mixup_hu(
                    starting_epoch, model, testloader,
                    testaugloader, test_criterion,
                    ENV,
                    logger)


        elif args.aug_method == "mixupthree":
            vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist, conflist, lossclean_list = test_three_mixup(
                starting_epoch, model, testloader,
                testloader, test_criterion,
                ENV,
                logger)

        else:
            if feature_mixup:
                vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist, conflist, lossclean_list = test_feature_mixup(
                    starting_epoch, model, testloader,
                    testshuffleloader, test_criterion,
                    ENV,
                    logger)

            else:
                vc_acc, vc_loss, phylist, losslist, entrlist, m_entrlist ,conflist, lossclean_list = test_mixup(starting_epoch, model, testloader,
                                                                                  testshuffleloader, test_criterion,
                                                                                  ENV,
                                                                                  logger)

        if args.query_mode == "single":
            folder = "phy"
        elif args.query_mode == "multiple":
            folder = "phy_multi"
        else:
            folder = "phy_tar"
        if (aug_type == "trades" or aug_type == "pgdat") and args.epsilon < 8:
            save_path = os.path.join(args.exp_name, folder, args.dataset, aug_type + "_" + str(args.epsilon))
        else:
            save_path = os.path.join(args.exp_name, folder, args.dataset, aug_type)
        if args.without_base:
            save_path = save_path + "_none"

        utils.create_path(save_path)
        if args.query_mode == "multiple":
            phylist = phylist.reshape(-1, 10)

        save_path = os.path.join(args.exp_name, folder, args.dataset, args.training_model)


        print(save_path)
        print(save_path)

        #print(os.path.join(save_path, "%s_%s_%s.npy" % ("phy_diff", str(index),str(LAM))))
        if feature_mixup:
            np.save(os.path.join(save_path,
                                 "%s_%s_layer%s_%s_%s.npy" % (args.aug_method, "phydiff",str(LAYER),str(index), str(int(LAM * 100)))),
                    phylist)

            np.save(os.path.join(save_path,
                                 "%s_%s_layer%s_%s_%s.npy" % (args.aug_method, "lossdiff",str(LAYER), str(index), str(int(LAM * 100)))),
                    losslist)
        else:

            np.save(os.path.join(save_path, "%s_%s_%s_%s_%s.npy" % (args.aug_method ,"HUphydiff", str(index),str(LAM),str(args.num_model))), phylist)

            np.save(os.path.join(save_path, "%s_%s_%s_%s_%s.npy" % (args.aug_method ,"HUlossdiff", str(index),str(LAM),str(args.num_model))), losslist)

        #np.save(os.path.join(save_path, "%s_%s_%s_%s.npy" % (str(args.num_model),args.aug_method ,"entrdiff", str(index),str(int(LAM*100)))), entrlist)

        #np.save(os.path.join(save_path, "%s_%s_%s_%s.npy" % (str(args.num_model),args.aug_method ,"mentrdiff", str(index), str(int(LAM * 100)))), m_entrlist)

            np.save(os.path.join(save_path, "%s_%s_%s_%s.npy" % (args.aug_method, "conf", str(index), str(LAM))), conflist)
        #np.save(os.path.join(save_path, "%s_%s_%s_%s.npy" % (args.aug_method, "loss", str(index), str(int(LAM * 100)))), lossclean_list)
    else:


        vc_acc, vc_loss = test_mixup(starting_epoch, model, testloader, test_criterion, ENV, logger)
    logger.info('Current loss: %.4f' % (vc_loss))
    logger.info('Current accuracy: %.2f' % (vc_acc))

    utils.delete_logger(name="resnet18_" + str(index), logger=logger)
    return ENV['best_acc']


if __name__ == "__main__":
    if args.dataset == "cifar10":
        config_path = "configs/config_10.json"
    elif args.dataset == "cifar100":
        config_path = "configs/config_100.json"
    elif args.dataset == "svhn":
        config_path = "configs/svhn.json"
    elif args.dataset == "purchase":
        config_path = "configs/purchase.json"
    elif args.dataset == "locations":
        config_path = "configs/locations.json"

    with open(config_path) as f:
        cfg = json.load(f)

    for j in range(args.s_model, args.t_model):
        print(j, ", ok")
        main(cfg, aug_type=args.aug_type, index=j, aug_index=0)
