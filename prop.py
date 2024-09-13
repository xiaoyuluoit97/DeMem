import numpy as np
import mem_luo
import os, json, time
import argparse
from util_frl import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import copy
import datetime
import matplotlib.pyplot as plt
import utils
from torch.nn.utils import clip_grad_norm_
import random
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
from models import *
from models.AlexNet import AlexNet
from dataset import get_loaders, root
from advtrain import cal_adv, cal_adv_dp
from trades_awp import AdvWeightPerturb, TradesAWP
from opacus.utils.batch_memory_manager import BatchMemoryManager
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
NOT_CONVERAGE_LIST = []
NUM_CLASSES = 10

configs = {
    'epsilon': 8 / 255,
    'num_steps': 10,
    'step_size': 2 / 255,
    'clip_max': 1,
    'clip_min': 0
}

configs1 = {
    'epsilon': 8 / 255,
    'num_steps': 20,
    'step_size': 2 / 255,
    'clip_max': 1,
    'clip_min': 0
}


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

    parser.add_argument('--noise', action='store_true', default=False)

    parser.add_argument('--FAT', action='store_true', default=False)
    parser.add_argument('--REG', action='store_true', default=False)
    parser.add_argument('--FLR', action='store_true', default=False)
    parser.add_argument('--MEM', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--dataset', default='cifar100',
                        choices=["cifar10", "cifar100", "svhn", "purchase", "locations"])
    parser.add_argument('--training_model', default='resnet18',
                        choices=["alexnet", "resnet18"])
    parser.add_argument('--cuda', default=0, type=int, help='perturbation bound')
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation bound')

    parser.add_argument('--s_model', default=0, type=int, help='the index of the first model')
    parser.add_argument('--t_model', default=1, type=int, help='the index of the last model')

    parser.add_argument('--reg_alpha', default=0.2, type=float, help='the index of the first model')

    parser.add_argument('--noise_multiplier', default=0.1, type=float, help='the index of the first model')
    parser.add_argument('--max_grad_norm', default=10, type=int, help='the index of the last model')

    parser.add_argument('--num_model', default=None, type=str, help='number of target model')
    parser.add_argument('--aug_type', default="base", type=str, help='aug type')
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--mode', default="train", choices=["all", "train", "target", "eval", "rob_acc"])
    parser.add_argument('--query_mode', default="single", choices=["single", "multiple", "white"])
    parser.add_argument('--without_base', action='store_true', default=False)
    parser.add_argument('--suffix', default="")
    parser.add_argument('--cnn', action='store_true', default=False)

    return parser.parse_args()


args = get_arguments()

MAX_GRAD_NORM = 1.5
EPSILON = 50.0
DELTA = 1e-6
MAX_PHYSICAL_BATCH_SIZE = 256
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda}')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def adjust_learning_rate(optimizer, epoch, allepoch=100):
    if allepoch == 50:
        if epoch >= 0.5 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.75 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
    ####0
    elif allepoch == 100 and args.dataset == "svhn":
        if epoch >= 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001
        elif epoch >= 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.0001
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.000001


    elif allepoch == 151 and args.FAT == True:
        if epoch >= 65:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 110:  # 130 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 130:
        if epoch >= 75:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 120:  # 130 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 110 and args.FAT == True:
        if epoch >= 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 105:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001



    elif allepoch == 130 and args.FAT == True:
        if epoch >= 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001


    elif allepoch == 120 and args.FAT == True:
        if epoch >= 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 110:  # 130 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 120 and args.aug_type == "trades_flr":
        if epoch >= 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01

    elif allepoch == 2:
        if epoch >= 75:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 130:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 120:
        if epoch >= 70:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 110:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 158:
        if epoch >= 75:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 130:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 300:
        if epoch >= 70:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 180:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001
        elif epoch >= 250:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.0005

    elif allepoch == 180 and args.FAT == True:
        if epoch >= 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 100 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 150 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.005


    elif allepoch == 200:
        if epoch >= 85:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
        elif epoch >= 180:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.001

    elif allepoch == 130 and args.FAT == True:
        if epoch >= 0.625 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.9 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01

    elif allepoch == 130:
        if epoch >= 0.625 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.85 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01

    elif allepoch == 100:
        if epoch >= 0.75 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.9 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01


    elif allepoch == 150:
        if epoch >= 0.75 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.9 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
    else:
        exit(0)


def SoftLabelNLL(predicted, target, reduce=False):
    if reduce:
        return -(target * predicted).sum(dim=1).mean()
    else:
        return -(target * predicted).sum(dim=1)


def trades_adv(model,
               x_natural,
               weight,
               epsilon,
               clip_max,
               clip_min,
               num_steps,
               step_size):
    # define KL-loss
    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)

    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model.base_forward(x_adv), dim=1),
                                   F.softmax(model.base_forward(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - new_eps), x_natural + new_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def TRADES_loss(adv_logits, natural_logits, loss_natural, beta, target):
    criterion_kl = nn.KLDivLoss(size_average=False).to(device)
    loss_robust = (1.0 / len(target)) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def test(epoch, model, testloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 100
    to_save = []
    for batch_idx, batch in enumerate(testloader):
        start = time.time()
        imgs, cids = batch
        imgs, cids = imgs.to(device), cids.to(device)

        if args.save_results:
            with torch.no_grad():
                pred = model.base_forward(imgs)
            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            # correspodinng correct labels predict value
            phy = torch.log(logits[iss, cids[iss]])
            pred = F.log_softmax(logits, dim=1)
            logits[iss, cids[iss]] = 0
            phy = phy - torch.log(torch.sum(logits, dim=1) + 1e-20)
            phy = phy.cpu().numpy()
            to_save.append(phy)
        else:
            with torch.no_grad():
                # imgs = cal_adv(model, criterion, "pgdat", imgs, cids, eps=args.epsilon)
                pred = model(imgs)
                # tcriterion = nn.NLLLoss()
                # imgs = cal_adv(model, tcriterion, "pgdat", imgs, cids, eps = 8)
                # model.eval()
                # pred = model(imgs)
                # loss = criterion(pred, cids)

        loss = criterion(pred, cids)

        _, predicted = pred.max(1)

        category_total += cids.size(0)
        category_loss += loss.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        if (batch_idx + 1) % log_frequency == 0:
            log_payload = {"category acc": 100. * (category_correct / category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                        **log_payload)
            logger.info(display)
    if args.save_results:
        phylist = np.concatenate(to_save)
        # 置信度评分
        return 100. * (category_correct / category_total), category_loss / category_total, phylist

    return 100. * (category_correct / category_total), category_loss / category_total


def test_save_inf(epoch, model, testloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    entr_to_save = []
    conf_to_save = []

    for batch_idx, batch in enumerate(testloader):
        imgs, cids = batch
        imgs = imgs.clone()
        imgs, cids = imgs.to(device), cids.to(device)
        if args.query_mode == "multiple":
            imgs_2 = imgs.clone().flip(4)
            imgs = torch.cat([imgs, imgs_2], dim=1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.save_results:
            with torch.no_grad():
                # pred_baseforward = model.base_forward(imgs)
                log_softmax, pred = model(imgs, require_logits=True)

            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            losses = torch.zeros(pred.size(0))
            m_entres = torch.zeros(pred.size(0))

            entr_logits = logits.cpu()
            for i in range(pred.size(0)):
                loss = -criterion(pred[i], cids[i])
                losses[i] = loss.item()
                # m_entres[i] = _m_entr_comp(entr_logits[i], cids[i]).item()

            conf = logits[iss, cids[iss]]
            phy = torch.log(logits[iss, cids[iss]])
            pred = F.log_softmax(logits, dim=1)
            logits[iss, cids[iss]] = 0

            phy = phy - torch.log(torch.sum(logits, dim=1) + 1e-20)
            phy = phy.cpu().numpy()
            losses = losses.cpu().numpy()
            m_entres = m_entres.cpu().numpy()
            entr_to_save.append(m_entres)
            loss_to_save.append(losses)
            to_save.append(phy)
            conf = conf.cpu().numpy()
            conf_to_save.append(conf)


        else:
            with torch.no_grad():
                pred = model(imgs)

        loss = criterion(pred, cids)

        _, predicted = pred.max(1)

        category_total += cids.size(0)
        category_loss += loss.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        if (batch_idx + 1) % log_frequency == 0:
            log_payload = {"category acc": 100. * (category_correct / category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                        **log_payload)
            logger.info(display)
    if args.save_results:
        conflist = np.concatenate(conf_to_save)
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        entrlist = np.concatenate(entr_to_save)

        return 100. * (
                    category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist, conflist

    return 100. * (category_correct / category_total), category_loss / category_total

def test_prop(epoch, model, testloader, criterion, ENV, logger):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    loss_to_save = []
    entr_to_save = []
    conf_to_save = []

    for batch_idx, batch in enumerate(testloader):
        imgs, cids = batch
        imgs = imgs.clone()
        imgs, cids = imgs.to(device), cids.to(device)
        start = time.time()

        if args.save_results:
            with torch.no_grad():
                # pred_baseforward = model.base_forward(imgs)
                log_softmax, pred = model(imgs, require_logits=True)

            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            losses = torch.zeros(pred.size(0))
            m_entres = torch.zeros(pred.size(0))

            entr_logits = logits.cpu()
            for i in range(pred.size(0)):
                loss = -criterion(pred[i], cids[i])
                losses[i] = loss.item()
                # m_entres[i] = _m_entr_comp(entr_logits[i], cids[i]).item()

            conf = logits[iss, cids[iss]]
            phy = torch.log(logits[iss, cids[iss]])
            pred = F.log_softmax(logits, dim=1)
            logits[iss, cids[iss]] = 0

            phy = phy - torch.log(torch.sum(logits, dim=1) + 1e-20)
            phy = phy.cpu().numpy()
            losses = losses.cpu().numpy()
            m_entres = m_entres.cpu().numpy()
            entr_to_save.append(m_entres)
            loss_to_save.append(losses)
            to_save.append(phy)
            conf = conf.cpu().numpy()
            conf_to_save.append(conf)


        else:
            with torch.no_grad():
                pred = model(imgs)

        loss = criterion(pred, cids)

        _, predicted = pred.max(1)

        category_total += cids.size(0)
        category_loss += loss.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        if (batch_idx + 1) % log_frequency == 0:
            log_payload = {"category acc": 100. * (category_correct / category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                        **log_payload)
            logger.info(display)
    if args.save_results:
        conflist = np.concatenate(conf_to_save)
        phylist = np.concatenate(to_save)
        losslist = np.concatenate(loss_to_save)
        entrlist = np.concatenate(entr_to_save)

        return 100. * (
                    category_correct / category_total), category_loss / category_total, phylist, losslist, entrlist, conflist

    return 100. * (category_correct / category_total), category_loss / category_total

def test_over_class(epoch, model, testloader, criterion, ENV, logger, type):
    logger.info("=" * 20 + "Test Epoch %d" % (epoch) + "=" * 20)
    model.eval()
    category_loss = 0
    category_correct = 0
    category_total = 0
    class_correct = [0] * NUM_CLASSES  # 增加一个列表来存储每个类别的正确预测数量
    class_total = [0] * NUM_CLASSES  # 增加一个列表来存储每个类别的样本总数
    log_frequency = 50
    to_save = []

    for batch_idx, batch in enumerate(testloader):
        start = time.time()
        imgs, cids = batch
        imgs, cids = imgs.to(device), cids.to(device)
        if args.save_results:
            with torch.no_grad():
                pred = model.base_forward(imgs)
            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            phy = torch.log(logits[iss, cids[iss]])
            pred = F.log_softmax(logits, dim=1)
            logits[iss, cids[iss]] = 0
            phy = phy - torch.log(torch.sum(logits, dim=1) + 1e-20)
            phy = phy.cpu().numpy()
            to_save.append(phy)
        else:
            with torch.no_grad():
                if type == "rob":
                    imgs_adv = cal_adv_dp(model, criterion, "pgdat", imgs, cids, test=True, eps=args.epsilon)
                    pred = model(imgs_adv)
                else:
                    pred = model(imgs)

        loss = criterion(pred, cids)
        _, predicted = pred.max(1)

        category_total += cids.size(0)
        category_loss += loss.item()
        category_correct += predicted.eq(cids).sum().item()

        for prediction, target in zip(predicted, cids):
            class_correct[target] += prediction.eq(target).item()
            class_total[target] += 1

        end = time.time()
        time_used = end - start
        if (batch_idx + 1) % log_frequency == 0:
            log_payload = {"category acc": 100. * (category_correct / category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                        **log_payload)
            logger.info(display)

    class_accuracy = [(class_correct[i] / class_total[i]) for i in range(NUM_CLASSES)]

    return 100. * (category_correct / category_total), category_loss / category_total, class_accuracy

def main(cfg, aug_type="none", index=0, aug_index=0):
    if args.dataset == "cifar10":
        if args.cnn:
            model = CNN(num_classes=10, channels=3).to(device)
        else:
            model = ResNet18().to(device)
    elif args.dataset == "cifar100":
        if args.training_model == "resnet18":
            model = ResNet18(num_classes=100).to(device)
        elif args.training_model == "alexnet":
            model = AlexNet(num_classes=100).to(device)
            model.apply(initialize_weights)
    elif args.dataset == "svhn":
        model = CNN(num_classes=10, channels=3).to(device)
    elif args.dataset == "purchase":
        model = MLP(num_classes=100, size=600).to(device)
    elif args.dataset == "locations":
        model = MLP(num_classes=30, size=446).to(device)

    if aug_type == "TradesAWP" or "AWP":
        proxy = copy.deepcopy(model)

    assert (aug_type in cfg['training_augmentations'])
    if aug_type in ['distillation', 'smooth', 'mixup']:
        criterion = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    else:
        if args.FAT or args.REG:
            criterion = nn.NLLLoss(reduction="none")
        else:
            criterion = nn.NLLLoss()

    test_criterion = nn.NLLLoss()

    if args.aug_type == "DP" or args.aug_type == "DP_pgdat":
        model = ModuleValidator.fix(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.dataset == "svhn":
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # , weight_decay=5e-4
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.exp_name == '':
        new_exp_name = 'exp_' + datetime.datetime.now()

    else:
        if (
                aug_type == "trades" or aug_type == "pgdat" or aug_type == "pgdat_fat" or aug_type == "pgdat_reg" or aug_type == "trades_fat" or aug_type == "trades_reg" or aug_type == "AWP_reg" or aug_type == "DP_pgdat"):
            if args.REG:
                new_exp_name = os.path.join(args.exp_name, args.dataset,
                                            aug_type + "_" + str(args.epsilon) + "_" + str(args.reg_alpha))
            else:
                new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type + "_" + str(args.epsilon))

        # or aug_type == "DP_pgdat"
        elif aug_type == "DP":
            new_exp_name = os.path.join(args.exp_name, args.dataset,
                                        aug_type + "_" + str(args.noise_multiplier) + "_" + str(args.max_grad_norm))

        else:
            new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type)

        if args.without_base:
            new_exp_name = new_exp_name + "_none"

    print(new_exp_name)
    print(new_exp_name)
    print(new_exp_name)
    if len(args.suffix) > 0:
        new_exp_name = os.path.join(new_exp_name, args.suffix)

    if args.mode == "all" or args.mode == "target":
        index = args.mode

    # if not args.train:
    # index = args.num_model

    exp_path = os.path.join(new_exp_name, args.training_model + "_" + str(index))
    log_file_path = os.path.join(exp_path, args.training_model + "_" + str(index))
    checkpoint_path = exp_path
    utils.create_path(checkpoint_path)

    logger = utils.setup_logger(name="resnet18_" + str(index), log_file=log_file_path + ".log")
    starting_epoch = 0

    # logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # logger.info("flops: %.4fG" % flops)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0}

    if aug_type == 'distillation':
        if args.dataset == "cifar10":
            if args.cnn:
                teacher = CNN(num_classes=10, channels=3).to(device)
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
        print(checkpoint_path.replace("distillation", "base"))
        tname = "none" if args.without_base else "base"
        utils.load_model(os.path.join(checkpoint_path.replace("distillation", tname), 'model'), teacher)
        teacher.eval()
    else:
        teacher = None

    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)
        if aug_type == "TradesAWP" or "AWP":
            proxy = torch.nn.DataParallel(proxy).to(device)

    if aug_type == "TradesAWP":
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.005)
    elif aug_type == "AWP" or aug_type == "AWP_fat" or aug_type == "AWP_reg":
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.01)
    else:
        awp_adversary = None

    if args.mode == "eval":
        bs = cfg['regular_batch_size'] * 4
    else:
        bs = cfg['regular_batch_size']

    if args.INFERENCE:
        if args.query_mode == "multiple":
            multiple = True
            bs = cfg['regular_batch_size']
        else:
            multiple = False
            bs = cfg['regular_batch_size'] * 4

        trainloader, testloader = get_loaders(args.dataset, aug_type, aug_index, cfg, args.training_model,
                                              shuffle=True, batch_size=bs,
                                              mode="inference", samplerindex=index,
                                              multiple=multiple,
                                              without_base=args.without_base)
    else:
        trainloader, testloader = get_loaders(args.dataset, aug_type, aug_index, cfg, args.training_model,
                                              shuffle=True, batch_size=bs,
                                              mode=args.mode, samplerindex=index,
                                              without_base=args.without_base)

    if args.aug_type == "DP" or args.aug_type == "DP_pgdat":
        privacy_engine = PrivacyEngine(secure_mode=False)

        model, optimizer, trainloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )

    if args.load_model:
        checkpoint, model = utils.load_model(os.path.join(checkpoint_path, 'model'), model)
        starting_epoch = checkpoint['epoch'] + 1

    if args.load_best_model:
        checkpoint, model = utils.load_model(os.path.join(checkpoint_path, 'model_best'), model)
        starting_epoch = checkpoint['epoch'] + 1

    logger.info("Starting Epoch: %d" % (starting_epoch))



    if args.save_results and args.INFERENCE:
            vc_acc, vc_loss, phylist, losslist, entrlist, conflist = test_save_inf(starting_epoch, model, testloader,
                                                                                   test_criterion, ENV, logger)

            if (
                    aug_type == "trades" or aug_type == "pgdat" or aug_type == "pgdat_fat" or aug_type == "pgdat_reg" or aug_type == "trades_fat" or aug_type == "trades_reg" or aug_type == "AWP_reg" or aug_type == "DP_pgdat"):
                if args.REG:
                    save_path = os.path.join(args.exp_name, "phy", args.dataset,
                                             aug_type + "_" + str(args.epsilon) + "_" + str(args.reg_alpha))
                elif args.FAT:
                    save_path = os.path.join(args.exp_name, "phy", args.dataset, aug_type + "_" + str(args.epsilon))
                else:
                    save_path = os.path.join(args.exp_name, "phy", args.dataset, aug_type + "_" + str(args.epsilon))
            elif aug_type == "DP" or aug_type == "DP_pgdat":
                save_path = os.path.join(args.exp_name, "phy", args.dataset,
                                         aug_type + "_" + str(args.noise_multiplier) + "_" + str(args.max_grad_norm))
            else:
                save_path = os.path.join(args.exp_name, "phy", args.dataset, aug_type)
            if args.load_best_model:
                save_path = save_path.replace("phy", "phy_best")

            print(save_path)

            if args.query_mode == "single":
                folder = "phy"
            elif args.query_mode == "multiple":
                folder = "phy_multi"
            else:
                folder = "phy_tar"

            if args.without_base:
                save_path = save_path + "_none"

            utils.create_path(save_path)
            if args.query_mode == "multiple":
                phylist = phylist.reshape(-1, 10)

            utils.create_path(save_path)
            print(save_path)
            print(save_path)
            print(os.path.join(save_path, "%s_%s.npy" % (folder, str(index))))

            np.save(os.path.join(save_path, "%s_%s.npy" % (folder, str(index))), phylist)

    else:
            print(testloader)

            # tc_acc, tc_loss, tc_class_acc = test_over_class(starting_epoch, model, trainloader, test_criterion, ENV, logger)
            if args.MEM:
                cle_vc_acc, vc_loss, vc_class_acc = test_over_mem(starting_epoch, model, testloader, test_criterion,
                                                                  ENV, logger)
                vc_class_acc_array = np.array(vc_class_acc)
                print(np.var(vc_class_acc_array))
                utils.create_path(f"acc/cifar100/clean/%s" % (aug_type))
                save_file_path = 'acc/cifar100/clean/%s/%s_cle_acc_mem.npy' % (aug_type, index)
                np.save(save_file_path, vc_class_acc_array)
                return cle_vc_acc, vc_class_acc
            else:
                train_vc_acc, _, _ = test_over_class(starting_epoch, model, trainloader, test_criterion, ENV, logger,
                                                     "cle")
                rob_vc_acc, vc_loss, vc_class_acc = test_over_class(starting_epoch, model, testloader, test_criterion,
                                                                    ENV, logger, "rob")
                cle_vc_acc, _, _ = test_over_class(starting_epoch, model, testloader, test_criterion, ENV, logger,
                                                   "cle")

                return train_vc_acc, rob_vc_acc, cle_vc_acc


    # logging.shutdown()
    utils.delete_logger(name="resnet18_" + str(index), logger=logger)
    torch.cuda.empty_cache()
    return vc_acc, vc_acc  # ENV['best_acc']


def cal_ic(data):
    std_dev = np.std(data, ddof=1)

    standard_error = std_dev / np.sqrt(len(data))
    return 1.96 * standard_error


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

    results = []
    for i in range(args.s_model, args.t_model):
        # for i in need_to_retrain:
        # if i in exists:
        # continue
        # try:
        if not args.train and not args.INFERENCE and not args.MEM:
            print("in it")
            tri, rob, cle = main(cfg, aug_type=args.aug_type, index=i)

        elif not args.train:
            tc_acc, vc_acc = main(cfg, aug_type=args.aug_type, index=i)

        elif args.MEM:
            _, _ = main(cfg, aug_type=args.aug_type, index=i)
        else:
            tc_acc, vc_acc = main(cfg, aug_type=args.aug_type, index=i)
            torch.cuda.empty_cache()
            # while vc_acc < 20:
            # print(f"now retrain model %s" % (str(i)))
            # tc_acc, vc_acc = main(cfg, aug_type=args.aug_type, index=i)

    np.save("tradesfat_notconverage.npy", NOT_CONVERAGE_LIST)

    # np.save(f"scala_figure/%s_robacc.npy"%(args.aug_type + "_" + str(args.epsilon) + "_" + str(args.reg_alpha)),rob_vc)
    # np.save(f"{args.aug_type}_notconverage.npy", NOT_CONVERAGE_LIST)
    if not args.train and not args.INFERENCE:
        output = f"aug_type {args.aug_type} ,epsilon {args.epsilon},train {np.mean(tri_vc):.4f},ic {cal_ic(tri_vc):.4f} , rob {np.mean(rob_vc):.4f}, ic {cal_ic(rob_vc):.4f}, cle {np.mean(cle_vc):.4f}, ic {cal_ic(cle_vc):.4f}\n"

        # 打开文件进行写入
        with open('all_accuracy.txt', 'a') as file:  # 'a' 模式表示追加内容到文件末尾
            file.write(output)