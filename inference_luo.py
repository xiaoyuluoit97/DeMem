from audioop import mul
import numpy as np

import os, json, time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import datetime
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
import utils
from models import ResNet18, MLP, CNN
from dataset import get_loaders, root
from advtrain import cal_adv


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
    parser.add_argument('--dataset', default = 'cifar100', choices=["cifar10", "cifar100", "svhn", "purchase", "locations"])
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation bound')
    parser.add_argument('--cuda', default=1, type=int, help='perturbation bound')
    parser.add_argument('--s_model', default=0, type=int, help='the index of the first model')
    parser.add_argument('--t_model', default=1, type=int, help='the index of the last model')
    parser.add_argument('--aug_type', default="base", type=str, help='aug type')
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--mode', default="train", choices=["all", "train", "target", "eval"])
    parser.add_argument('--query_mode', default="single", choices=["single", "multiple", "white"])
    parser.add_argument('--training_model', default="resnet18", choices=["resnet18", "alexnet"])
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
        return -(target * predicted).sum(dim=1).mean()
    else:
        return -(target * predicted).sum(dim=1)

def train(epoch, model, optimizer, trainloader, ENV, aug_type, criterion, cfg, teacher, logger, aug_index = 0):
    logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
    model.train()
    log_frequency = 50
    category_loss= 0
    category_correct = 0
    category_total = 0
    criterion_kl = nn.KLDivLoss(reduction='sum')
    if aug_type == "mixup":
        iterator = zip(trainloader, trainloader)
    else:
        iterator = enumerate(trainloader)
    for batch in iterator:
        start = time.time()
        if aug_type == "mixup":
            (imgs, cids), (imgs_2, cids_2) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)
        else:
             _, (imgs, cids) = batch
             imgs, cids = imgs.to(device), cids.to(device)
        optimizer.zero_grad()

        if aug_type == "distillation":
            T = cfg['augmentation_params']['distillation'][aug_index]
            dt = teacher.forward_w_temperature(imgs, T).detach()
            pred = model(imgs)
            loss = criterion(pred, dt)
        elif aug_type == "smooth":
            pred = model(imgs)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids.view(-1, 1), 1)

            smoothing_coef = cfg['augmentation_params']['smooth'][aug_index]
            b_y_one_hot = (1-smoothing_coef)*b_y_one_hot + (smoothing_coef/model.num_classes)
            
            loss = criterion(pred, b_y_one_hot)
        elif aug_type == "mixup":
            alpha = cfg['augmentation_params']['mixup'][aug_index]
            lam = np.random.beta(alpha, alpha)
            b_x = (lam * imgs) + ((1 - lam) * imgs_2)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids.view(-1, 1), 1)
            b_y_one_hot_2 = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids_2.view(-1, 1), 1)

            b_cid = (lam * b_y_one_hot) + ((1 - lam) * b_y_one_hot_2)
            pred = model(b_x)
            loss = criterion(pred, b_cid)
        elif aug_type == "disturblabel":
            C = model.num_classes
            alpha = cfg['augmentation_params']['disturblabel'][aug_index]
            p_c = (1 - ((C - 1)/C) * alpha)
            p_i = (1 / C) * alpha
            b_y = cids.view(-1, 1)   # batch y

            b_y_one_hot = (torch.ones(b_y.shape[0], C) * p_i).to(device)
            b_y_one_hot.scatter_(1, b_y, p_c)
            b_y_one_hot = b_y_one_hot.view( *(tuple(cids.shape) + (-1,) ) )

            # sample from Multinoulli distribution
            distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
            b_y_disturbed = distribution.sample()
            b_y_disturbed = b_y_disturbed.max(dim=1)[1]  # back to categorical
            pred = model(imgs)
            loss = criterion(pred, b_y_disturbed)
        elif aug_type == "trades":
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            model.train()
            pred, logits = model(imgs, require_logits = True)
            loss_natural = criterion(pred, cids)
            pred_adv = model(imgs_adv)
            loss_robust = (1.0 / (pred.shape[0])) * criterion_kl(pred_adv, F.softmax(logits, dim = 1))
            loss_natural = criterion(pred, cids)
            loss = loss_natural + 6 * loss_robust
        elif aug_type == "pgdat":
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            model.train()
            pred = model(imgs_adv)
            loss = criterion(pred, cids)
        else:
            pred = model(imgs)
            loss = criterion(pred, cids)

        loss.backward()
        optimizer.step()
        
        category_loss += loss.item()
        _, predicted = pred.max(1)
        category_total += cids.size(0)
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        
        if ENV["global_step"] % log_frequency == 0:
            log_payload = {"loss": category_loss/category_total, "acc": 100.*(category_correct/category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                           **log_payload)
            logger.info(display)

        ENV["global_step"] += 1
    return category_loss/category_total, 100.*(category_loss/category_total)

def _log_value(probs, small_value=1e-30):
    return -np.log(np.maximum(probs, small_value))

def _m_entr_comp(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[true_labels] = reverse_probs[true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[true_labels] = log_probs[true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs))

def _m_entr_compsssssssss(probs, true_label):
    log_probs = _log_value(probs)
    reverse_probs = 1-probs

    one_mins_pro = reverse_probs[true_label]
    log_pro = log_probs[true_label]

    first_part = np.multiply(one_mins_pro, log_pro)

    log_reverse_probs = _log_value(reverse_probs)
    second_log = torch.cat((log_reverse_probs[:true_label], log_reverse_probs[true_label + 1:]))
    second_prob = torch.cat((probs[:true_label], probs[true_label + 1:]))

    second_part = np.multiply(second_prob, second_log)



    return np.sum(first_part,second_part)

def test(epoch, model, testloader, criterion, ENV, logger):
    logger.info("="*20 + "Test Epoch %d" % (epoch) + "="*20)
    model.eval()
    category_loss= 0
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
            imgs = torch.cat([imgs, imgs_2], dim = 1)
            bs, ncrops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            cids = cids.reshape(imgs.shape[0] // 10, 1).repeat(1, 10).reshape(-1)

        start = time.time()

        if args.save_results:
            with torch.no_grad():
                pred = model.base_forward(imgs)
            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            losses = torch.zeros(pred.size(0))
            m_entres = torch.zeros(pred.size(0))

            entr_logits = logits.cpu()
            for i in range(pred.size(0)):
                loss = -criterion(pred[i], cids[i])
                losses[i] = loss.item()
                m_entres[i] = _m_entr_comp(entr_logits[i], cids[i]).item()

            conf = logits[iss, cids[iss]]
            phy = torch.log(logits[iss,cids[iss]])
            pred = F.log_softmax(logits, dim = 1)
            logits[iss,cids[iss]] = 0


            phy = phy - torch.log(torch.sum(logits,dim=1) + 1e-20)
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
        if (batch_idx+1) % log_frequency == 0:
            log_payload = {"category acc": 100.* (category_correct/category_total)}
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


        return 100.* (category_correct/category_total), category_loss / category_total, phylist,losslist,entrlist,conflist

    return 100.* (category_correct/category_total), category_loss / category_total


def main(cfg, aug_type = "none", index = 0, aug_index = 0):
    if args.dataset == "cifar10":
        if args.cnn:
            model = CNN(num_classes=10).to(device)
        else:
            model = ResNet18().to(device)
    elif args.dataset == "cifar100":
        model = ResNet18(num_classes=100).to(device)
    elif args.dataset == "svhn":
        model = CNN(num_classes=10).to(device)
    elif args.dataset == "purchase":
        model = MLP(num_classes=100, size=600).to(device)
    elif args.dataset == "locations":
        model = MLP(num_classes=30, size=446).to(device)
    assert(aug_type in cfg['training_augmentations'])
    if aug_type in ['distillation', 'smooth', 'mixup']:
        criterion = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    else:
        criterion = nn.NLLLoss()
    test_criterion = nn.NLLLoss()

    if args.aug_type == "DP" or args.aug_type == "DP2":
        model = ModuleValidator.fix(model)


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


    #index = "target"


    exp_path = os.path.join(new_exp_name, "resnet18_" + str(index))
    log_file_path = os.path.join(exp_path, "resnet18_" + str(index))
    checkpoint_path = exp_path
    utils.create_path(checkpoint_path)

    logger = utils.setup_logger(name="resnet18_" + str(index), log_file=log_file_path + ".log")
    starting_epoch = 0

    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = { 'global_step': 0,
            'best_acc': 0.0,
            'curren_acc': 0.0,
            'best_pgd_acc': 0.0}



    if args.load_model:
        checkpoint , model = utils.load_model(os.path.join(checkpoint_path, 'model'), model)
        starting_epoch = checkpoint['epoch'] + 1
    if args.load_best_model:
        checkpoint , model = utils.load_model(os.path.join(checkpoint_path, 'model'), model)
        starting_epoch = checkpoint['epoch'] + 1


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
    trainloader, testloader = get_loaders(args.dataset, aug_type, aug_index, cfg,args.training_model,
                                            shuffle=True, batch_size=bs, 
                                            mode = args.mode, samplerindex = index, 
                                            multiple = multiple,
                                            without_base = args.without_base)

    logger.info("Starting Epoch: %d" % (starting_epoch))
    
    if args.save_results:
        vc_acc, vc_loss, phylist, losslist, entrlist, conflist = test(starting_epoch, model, testloader, test_criterion, ENV, logger)
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

        print(os.path.join(save_path, "%s_%s.npy" % (folder, str(index))))

        np.save(os.path.join(save_path, "%s_%s.npy" % (folder, str(index))), phylist)

        #np.save(os.path.join(save_path, "%s_%s.npy" % ("loss", str(index))), losslist)

        #np.save(os.path.join(save_path, "%s_%s.npy" % ("mentr", str(index))), entrlist)

        #np.save(os.path.join(save_path, "%s_%s.npy" % ("conf", str(index))), conflist)
    else:
        vc_acc, vc_loss = test(starting_epoch, model, testloader, test_criterion, ENV, logger)
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
        main(cfg, aug_type = args.aug_type, index = j, aug_index = 0)



    
    