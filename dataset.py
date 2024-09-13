import numpy as np
import os
import torch
import json
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from torch.utils.data import Subset, SubsetRandomSampler, Sampler
from utils import *
import random
import matplotlib.pyplot as plt
import torch

#root = "/home/lixiao/data2/privacy_and_aug"
root = "/data/luo/reproduce/privacy_and_aug" # for tifs major revision
#root = "/Users/luo/reproduce/privacy_and_aug"
#https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py


ANAMEM = False
MIX_HU = False
MIX_DIS = False
MIX_CUT = False

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, device):
        self.n_holes = n_holes
        self.length = length
        self.device = device


    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if img.ndim == 4: # batch
            h = img.size(2)
            w = img.size(3)
        else: # single img
            h = img.size(1)
            w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            
        mask = torch.from_numpy(mask).to(self.device)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class MyTransforms(object):
    def __init__(self, net, inputs_sample, device):
        """
        :param net:
        :param inputs_sample: N_label * Num_per_label * C * W * H
        :param device:
        """
        self.net = net
        self.inputs_sample = inputs_sample
        self.device = device

    def composed_trans(self, trans_list, x, y, ids=0):
        for trans in trans_list:
            x = trans(x, y, ids=ids)
        return x

    def mix_up(self, x, y, r=0.5, ids=0):
        img = self.inputs_sample[y.cpu()][:, ids].to(self.device)
        return x * r + img * (1 - r)

    def mix_up_dis(self, x, y, r=0.7, ids=0):
        img = self.inputs_sample[y.cpu()][:, ids].to(self.device)
        mask = torch.zeros_like(x).uniform_() < r
        return torch.where(mask, x, img)

    def cut_mix(self, x, y, cut_ratio=0.7, ids=0):
        img = self.inputs_sample[y.cpu()][:, ids].to(self.device)
        mask = torch.zeros_like(x).bool()
        w = x.shape[-1]
        dw = int(w * cut_ratio)
        hstart = (torch.randint((w - dw + 1), x.shape[:1], device=self.device)[:, None, None, None] + torch.arange(dw, device=self.device)[None, None, :, None]).expand(-1, x.shape[1], -1, x.shape[3])
        wstart = (torch.randint((w - dw + 1), x.shape[:1], device=self.device)[:, None, None, None] + torch.arange(dw, device=self.device)[None, None, None, :]).expand(-1, x.shape[1], dw, -1)
        a = mask.new_zeros([x.shape[0], dw, x.shape[2], x.shape[3]])
        b = mask.new_ones([x.shape[0], x.shape[1], dw, dw])
        a.scatter_(3, wstart, b)
        mask.scatter_(2, hstart, a)
        return torch.where(mask, x, img)


    def gaussian_noise(self, x, y=None, delta=0.05, ids=0):
        return x + x.new(x.shape).normal_() * delta

    def step_noise(self, x, y=None, delta=0.05, ids=0):
        return x + x.new(x.shape).normal_().sign() * delta



class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels, use_aug = False, multiple_query = False, resize=True,size=32, padding=4, device='cpu'):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.resize = resize
        self.size = size
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.use_aug = use_aug
        self.multiple_query = multiple_query
        self.transforms = None
        self.gaussian_std = None
        self.base_t = None

        self.padding = padding

        if self.resize:
            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.multiple_query:
            self.add_mutiplequery()
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]


        if self.resize:
            data = self.resize_transform(data)

        if self.multiple_query:
            data = self.multi_transforms(data)

        if self.use_aug and self.base_t is not None:
            data = self.base_t(data)

        if self.use_aug:
            if self.transforms is not None:
                data = self.transforms(data)
            
            if self.gaussian_std is not None:
                data = torch.clamp(data + torch.randn(data.size(), device=self.device) * self.gaussian_std, min=0, max=1)

        return (data, self.labels[idx])


    def add_base(self):
        self.base_t = transforms.Compose([transforms.ToPILImage(),
                    transforms.RandomCrop(self.size, padding=self.padding),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])

    def add_cutout(self, cutout_size):
        if cutout_size > 0:
            self.transforms = transforms.Compose([Cutout(n_holes=1, length=cutout_size, device=self.device)])

    def add_gaussian_aug(self, std_dev):
        self.gaussian_std = std_dev
    
    def add_jitter(self, jitter_param):
        self.transforms = transforms.Compose([transforms.ToPILImage(), 
                    transforms.ColorJitter(brightness=jitter_param, contrast = jitter_param, saturation = jitter_param, hue = jitter_param), 
                    transforms.ToTensor()])

    def add_mutiplequery(self):
        self.multi_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Pad(self.padding),
                    transforms.FiveCrop(self.size),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) # returns a 5D tensor [B, 5, C, H, W]
                    ])

    def to_alexnet(self, data):
        trans_to_alexnet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])
        data = trans_to_alexnet(data)
        return data


class ManualData_MEM(torch.utils.data.Dataset):
    def __init__(self, data, labels,mems, use_aug=False, multiple_query=False, resize=True, size=32, padding=4,
                 device='cpu'):

        self.data = torch.from_numpy(data).to(device, dtype=torch.float)

        self.resize = resize
        self.size = size
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.mems = torch.from_numpy(mems).to(device, dtype=torch.long)
        self.use_aug = use_aug
        self.multiple_query = multiple_query
        self.transforms = None
        self.gaussian_std = None
        self.base_t = None

        self.padding = padding

        if self.resize:
            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.multiple_query:
            self.add_mutiplequery()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.resize:
            data = self.resize_transform(data)

        if self.multiple_query:
            data = self.multi_transforms(data)

        if self.use_aug and self.base_t is not None:
            data = self.base_t(data)

        if self.use_aug:
            if self.transforms is not None:
                data = self.transforms(data)

            if self.gaussian_std is not None:
                data = torch.clamp(data + torch.randn(data.size(), device=self.device) * self.gaussian_std, min=0,
                                   max=1)

        return (data, self.labels[idx],self.mems[idx])

    def add_base(self):
        self.base_t = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomCrop(self.size, padding=self.padding),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    def add_cutout(self, cutout_size):
        if cutout_size > 0:
            self.transforms = transforms.Compose([Cutout(n_holes=1, length=cutout_size, device=self.device)])

    def add_gaussian_aug(self, std_dev):
        self.gaussian_std = std_dev

    def add_jitter(self, jitter_param):
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param,
                                                                     saturation=jitter_param, hue=jitter_param),
                                              transforms.ToTensor()])

    def add_mutiplequery(self):
        self.multi_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(self.padding),
            transforms.FiveCrop(self.size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
            # returns a 5D tensor [B, 5, C, H, W]
        ])

    def to_alexnet(self, data):
        trans_to_alexnet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])
        data = trans_to_alexnet(data)
        return data


class ManualData_MIXUP(torch.utils.data.Dataset):
    def __init__(self, data, labels, use_aug=False, multiple_query=False, resize=True, size=32, padding=4,
                 device='cpu'):

        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.resize = resize
        self.size = size
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.use_aug = use_aug
        self.multiple_query = multiple_query
        self.transforms = None
        self.gaussian_std = None
        self.base_t = None
        self.padding = padding
        self.inputs_sample = self.group_by_label(self.data,self.labels)

        if self.resize:
            self.resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.multiple_query:
            self.add_mutiplequery()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if MIX_HU:
            data = self.add_mixup(self.data[idx],self.labels[idx])
        elif MIX_DIS:
            data = self.mix_up_dis(self.data[idx], self.labels[idx])
        elif MIX_CUT:
            data = self.cut_mix(self.data[idx], self.labels[idx])
        return (data, self.labels[idx])

    def add_base(self):
        self.base_t = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomCrop(self.size, padding=self.padding),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    def add_mixup(self, x, y, r=0.5):
        ids = random.randint(0, 599)
        img = self.inputs_sample[y.cpu()][ids,:].to(self.device)
        return x * r + img * (1 - r)

    def mix_up_dis(self, x, y, r=0.7):
        ids = random.randint(0, 599)
        img = self.inputs_sample[y.cpu()][ids,:].to(self.device)
        mask = torch.zeros_like(x).uniform_() < r
        return torch.where(mask, x, img)

    def cut_mix(self, x, y, cut_ratio=0.7):
        ids = random.randint(0, 599)
        img = self.inputs_sample[y.cpu()][ids,:].to(self.device)
        mask = torch.zeros_like(x).bool()
        w = x.shape[-1]
        dw = int(w * cut_ratio)
        hstart = (torch.randint((w - dw + 1), x.shape[:1], device=self.device)[:, None, None, None] + torch.arange(dw, device=self.device)[None, None, :, None]).expand(-1, x.shape[1], -1, x.shape[3])
        wstart = (torch.randint((w - dw + 1), x.shape[:1], device=self.device)[:, None, None, None] + torch.arange(dw, device=self.device)[None, None, None, :]).expand(-1, x.shape[1], dw, -1)

        a = mask.new_zeros([x.shape[0], dw, x.shape[2], x.shape[3]])
        b = mask.new_ones([x.shape[0], x.shape[1], dw, dw])
        a.scatter_(3, wstart, b)
        mask.scatter_(2, hstart, a)
        return torch.where(mask, x, img)


    def to_alexnet(self, data):
        trans_to_alexnet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])
        data = trans_to_alexnet(data)
        return data

    def group_by_label(self,data, labels):
        # 获取标签的独特值和每个标签的数量
        unique_labels = torch.unique(labels)
        N_label = len(unique_labels)
        Num_per_label = (labels == unique_labels[0]).sum().item()
        C, W, H = data.shape[1], data.shape[2], data.shape[3]

        # 初始化一个空张量来存储结果
        grouped_data = torch.zeros((N_label, Num_per_label, C, W, H), dtype=data.dtype)

        for i, label in enumerate(unique_labels):
            # 选择属于当前标签的所有数据
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            grouped_data[i] = data[label_indices]

        return grouped_data


class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



def get_loaders(ds_name, data_aug_type, aug_index, cfg,training_model, shuffle=True, batch_size=128,
                            device='cpu', mode = 'train', samplerindex = 0,
                                        use_aug = False, multiple = False, without_base = False):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 4

    if ds_name == "cifar10":
        datasets = get_cifar10_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "cifar100":
        datasets = get_cifar100_datasets(device=device, training_model=training_model,use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "svhn":
        datasets = get_svhn_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "purchase":
        datasets = get_purchase_dataset(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "locations":
        datasets = get_locations_dataset(device=device, use_aug = use_aug, multiple_query = multiple)

    train_ds, test_ds = datasets
    if data_aug_type != "none" and not without_base:
        train_ds.add_base()

    if mode == "inference":
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
        return train_loader,test_loader


    if data_aug_type == "noise":
        train_ds.add_gaussian_aug(cfg['augmentation_params']['noise'][aug_index])
    elif data_aug_type == "cutout":
        train_ds.add_cutout(cfg['augmentation_params']['cutout'][aug_index])
    elif data_aug_type == "jitter":
        train_ds.add_jitter(cfg['augmentation_params']['jitter'][aug_index])

    if mode == 'train':
        if ds_name == "locations":
            f = open("sampleinfo/samplelist_locations.txt", "r")
            samplelist = eval(f.read())
            f.close()
            size = 5010
        else:
            f = open("sampleinfo/samplelist.txt", "r")
            samplelist = eval(f.read())
            f.close()
            size = 60000
        inlist = set(samplelist[samplerindex])
        outlist = []
        for i in range(size):
            if i not in inlist:
                outlist.append(i)

        if ANAMEM:
            max_index = len(test_ds)  # 获取 test_ds 的最大索引范围
            outlist = [i for i in outlist if i < max_index]
            inlist = [i for i in inlist if i < max_index]

            train_ds = Subset(train_ds, indices = list(inlist))
            test_ds = Subset(test_ds, indices = outlist)
        else:
            train_ds = Subset(train_ds, indices = samplelist[samplerindex])
            test_ds = Subset(test_ds, indices = outlist)

        # samplelist = np.load("memtoacc/deletemem_0_%s.npy" % samplerindex)
        # samplelist = np.load("memtoacc/random_0_%s.npy" % samplerindex)
        # samplelist = list(samplelist)
        # print("ok: ", samplerindex)
        # IN = set(samplelist)
        # outlist = []
        # for i in range(60000):
        #     if i not in IN:
        #         outlist.append(i)
        # train_ds = Subset(train_ds, indices = samplelist)
        # test_ds = Subset(test_ds, indices = [i for i in range(50000,60000)])

    elif mode == "rob_acc":
        if ds_name == "locations":
            f = open("sampleinfo/samplelist_locations.txt", "r")
            samplelist = eval(f.read())
            f.close()
            size = 5010
        else:
            f = open("sampleinfo/samplelist.txt", "r")
            samplelist = eval(f.read())
            f.close()
            size = 60000

        inlist = set(samplelist[int(samplerindex)])
        outlist = []

        for i in range(size):
            if i not in inlist:
                outlist.append(i)

        if ANAMEM:
            max_index = len(test_ds)  # 获取 test_ds 的最大索引范围
            outlist = [i for i in outlist if i < max_index]
            inlist = [i for i in inlist if i < max_index]

        train_ds = Subset(train_ds, indices = list(inlist))
        test_ds = Subset(test_ds, indices = outlist)

    elif mode == "target":
        if ds_name == "locations":
            f = open("sampleinfo/target_locations.txt", "r")
            target = eval(f.read())
            f.close()
            size = 5010
        else:
            f = open("sampleinfo/target.txt", "r")
            target = eval(f.read())
            f.close()
            size = 60000
        inlist = set(target)
        outlist = []
        for i in range(size):
            if i not in inlist:
                outlist.append(i)

        if ANAMEM:
            max_index = len(test_ds)  # 获取 test_ds 的最大索引范围
            outlist = [i for i in outlist if i < max_index]
            inlist = [i for i in inlist if i < max_index]

        train_ds = Subset(train_ds, indices = list(inlist))
        test_ds = Subset(test_ds, indices = outlist)

    elif mode == "all":
        train_ds = Subset(train_ds, indices = [i for i in range(50000)])
        test_ds = Subset(test_ds, indices = [i for i in range(50000, 60000)])

    elif mode == "eval":
        eval_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle = False,  num_workers=num_workers)
        return  None, eval_loader

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = shuffle,  num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_loader, test_loader

def get_mixupmem_loaders(ds_name, data_aug_type, aug_index, cfg,training_model, shuffle=True, batch_size=128,mem=0, device='cpu', use_aug = False, multiple = False, without_base = False):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0

    if ds_name == "cifar10":
        datasets = get_cifar10_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "cifar100":
        datasets = get_cifar100_datasets(device=device, training_model=training_model,use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "svhn":
        datasets = get_svhn_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "purchase":
        datasets = get_purchase_dataset(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "locations":
        datasets = get_locations_dataset(device=device, use_aug = use_aug, multiple_query = multiple)

    train_ds, test_ds = datasets

    if data_aug_type != "none" and not without_base:
        train_ds.add_base()

    if data_aug_type == "noise":
        train_ds.add_gaussian_aug(cfg['augmentation_params']['noise'][aug_index])
    elif data_aug_type == "cutout":
        train_ds.add_cutout(cfg['augmentation_params']['cutout'][aug_index])
    elif data_aug_type == "jitter":
        train_ds.add_jitter(cfg['augmentation_params']['jitter'][aug_index])



    loaded_mem_equal_0 = np.load('sampleinfo/mem_equal_0.npy')
    loaded_mem_equal_1 = np.load('sampleinfo/mem_equal_1.npy')
    train_subset_0 = Subset(test_ds, loaded_mem_equal_0)
    train_subset_1 = Subset(test_ds, loaded_mem_equal_1)


    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if mem == 1:
        test_shufflemem_loader = torch.utils.data.DataLoader(train_subset_0, batch_size=batch_size, shuffle = False,  num_workers=num_workers,sampler=CustomSampler(loaded_mem_equal_0))

    elif mem == 0:
        test_shufflemem_loader = torch.utils.data.DataLoader(train_subset_1, batch_size=batch_size, shuffle=False,
                                                             num_workers=num_workers,
                                                             sampler=CustomSampler(loaded_mem_equal_1))


    return test_loader, test_shufflemem_loader

def get_mixupmem_loaders_hu(ds_name, data_aug_type, aug_index, cfg,training_model, shuffle=True, batch_size=128,mem=0, device='cpu', use_aug = False, multiple = False, without_base = False):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0

    if ds_name == "cifar10":
        datasets = get_cifar10_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "cifar100":
        datasets = get_cifar100_datasets(device=device, training_model=training_model,use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "svhn":
        datasets = get_svhn_datasets(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "purchase":
        datasets = get_purchase_dataset(device=device, use_aug = use_aug, multiple_query = multiple)
    elif ds_name == "locations":
        datasets = get_locations_dataset(device=device, use_aug = use_aug, multiple_query = multiple)

    test_ds, test_ds_aug = datasets

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader_aug = torch.utils.data.DataLoader(test_ds_aug, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return test_loader, test_loader_aug


def get_squ_loaders(ds_name, data_aug_type, aug_index, cfg, training_model,shuffle=True, batch_size=128,
                device='cpu', mode='train', samplerindex=0,
                use_aug=False, multiple=False, without_base=False):
    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0

    if ds_name == "cifar10":
        datasets = get_cifar10_datasets(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "cifar100":
        datasets = get_cifar100_datasets(device=device, training_model=training_model,use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "svhn":
        datasets = get_svhn_datasets(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "purchase":
        datasets = get_purchase_dataset(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "locations":
        datasets = get_locations_dataset(device=device, use_aug=use_aug, multiple_query=multiple)

    test_aug_ds, test_ds = datasets
    if data_aug_type != "none" and not without_base:
        test_ds.add_base()
    if data_aug_type == "noise":
        test_aug_ds.add_gaussian_aug(cfg['augmentation_params']['noise'][aug_index])
    elif data_aug_type == "cutout":
        test_aug_ds.add_cutout(cfg['augmentation_params']['cutout'][aug_index])
    elif data_aug_type == "jitter":
        test_aug_ds.add_jitter(cfg['augmentation_params']['jitter'][aug_index])



    test_loader_aug = torch.utils.data.DataLoader(test_aug_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    return test_loader, test_loader_aug


import torchvision.transforms as transforms
import os


def resize_cifar100_datasets(train_dataset, test_dataset, visualize=False):
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if visualize:
        visualize_resized_images(train_dataset, num_images=5)
    # Apply the transform to train dataset
    train_dataset.transform = transform
    # Apply the transform to test dataset
    test_dataset.transform = transform

    if visualize:
        visualize_resized_images(train_dataset, num_images=5)

    return train_dataset, test_dataset


def visualize_resized_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        img, label = dataset[i]
        #img = img * 0.5 + 0.5  # Unnormalize the image
        img = img.permute(1, 2, 0).numpy()  # Convert from tensor to numpy array
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

def get_shuffle_loaders(ds_name, data_aug_type, aug_index, cfg,training_model, shuffle=True, batch_size=128, device='cpu', use_aug=False, multiple=False, without_base=False):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0
    if ds_name == "cifar10":
        datasets = get_cifar10_datasets(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "cifar100":
        datasets = get_cifar100_datasets(device=device, training_model=training_model,use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "svhn":
        datasets = get_svhn_datasets(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "purchase":
        datasets = get_purchase_dataset(device=device, use_aug=use_aug, multiple_query=multiple)
    elif ds_name == "locations":
        datasets = get_locations_dataset(device=device, use_aug=use_aug, multiple_query=multiple)

    train_ds, test_ds = datasets
    if data_aug_type != "none" and not without_base:
        train_ds.add_base()

    if data_aug_type == "noise":
        train_ds.add_gaussian_aug(cfg['augmentation_params']['noise'][aug_index])
    elif data_aug_type == "cutout":
        train_ds.add_cutout(cfg['augmentation_params']['cutout'][aug_index])
    elif data_aug_type == "jitter":
        train_ds.add_jitter(cfg['augmentation_params']['jitter'][aug_index])

    test_ds = Subset(test_ds, indices=[i for i in range(60000)])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_shuffle_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)

    return test_loader, test_shuffle_loader




def get_cifar10_datasets(device='cpu', use_aug = False, multiple_query = False):
    create_path(os.path.join(root, "cifar10"))
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(os.path.join(root, "cifar10"), train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR10(os.path.join(root, "cifar10"), train=False, download=True, transform=t)
    
    # To Manual Data
    train_data, test_data = (train_dataset.data / 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2)) 
    all_data = np.concatenate([train_data, test_data], axis=0)
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    all_labels = np.concatenate([train_labels, test_labels])
    train_dataset = ManualData(all_data, all_labels, True, multiple_query = multiple_query, size = 32, padding=4, device=device)
    test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, size = 32, padding=4, device=device)
    return train_dataset, test_dataset

def get_cifar100_datasets(device='cpu', training_model= "alexnet",use_aug = False, multiple_query = False):
    create_path(os.path.join(root, "cifar100"))
    t = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100(os.path.join(root, "cifar100"), train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR100(os.path.join(root, "cifar100"), train=False, download=True, transform=t)

        # To Manual Data
    train_data, test_data = (train_dataset.data / 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2)) 
    all_data = np.concatenate([train_data, test_data], axis=0)
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    all_labels = np.concatenate([train_labels, test_labels])

    size = 32
    resize = False
    if training_model == "alexnet":
        size = 224
        resize = True
    if ANAMEM:
        mem = np.load("sampleinfo/cifar100_mem_21bin_labels.npy")

        train_dataset = ManualData_MEM(train_data, train_labels, mem, True, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)
        test_dataset = ManualData_MEM(train_data, train_labels, mem, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)

        return train_dataset, test_dataset

    elif MIX_HU or MIX_CUT or MIX_DIS:
        test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)
        test_dataset_aug = ManualData_MIXUP(all_data, all_labels, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)
        return test_dataset, test_dataset_aug

    else:
        train_dataset = ManualData(all_data, all_labels, True, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)
        test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)

        return train_dataset, test_dataset



def get_cifar100_mixup_datasets(device='cpu', training_model= "alexnet",use_aug = False, multiple_query = False):
    create_path(os.path.join(root, "cifar100"))
    t = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100(os.path.join(root, "cifar100"), train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR100(os.path.join(root, "cifar100"), train=False, download=True, transform=t)

        # To Manual Data
    train_data, test_data = (train_dataset.data / 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2))
    all_data = np.concatenate([train_data, test_data], axis=0)
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    all_labels = np.concatenate([train_labels, test_labels])

    size = 32
    resize = False
    if training_model == "alexnet":
        size = 224
        resize = True



    train_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)
    test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, resize=resize,size = size, padding=4, device=device)

    return test_dataset, test_dataset

def get_svhn_datasets(device='cpu', use_aug = False, multiple_query = False):
    create_path(os.path.join(root, "svhn"))
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.SVHN(os.path.join(root, "svhn"), split='train', download=True, transform=t)
    test_dataset = datasets.SVHN(os.path.join(root, "svhn"), split='test', download=True, transform=t)
    
    # To Manual Data
    train_data, test_data = (train_dataset.data / 255) , (test_dataset.data / 255)

    all_data = np.concatenate([train_data, test_data], axis=0)
    all_data = all_data[:60000]
    train_labels, test_labels = np.array(train_dataset.labels), np.array(test_dataset.labels)
    all_labels = np.concatenate([train_labels, test_labels])
    all_labels = all_labels[:60000]
    train_dataset = ManualData(all_data, all_labels, True, multiple_query = multiple_query, size = 32, padding=4, device=device)
    test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, size = 32, padding=4, device=device)
    return train_dataset, test_dataset




def get_locations_dataset(device='cpu', use_aug = False, multiple_query = False): # 30 categories, 446 dim
    create_path(os.path.join(root, "locations"))

    data = np.load(os.path.join(root, "locations", "dataset_locations.npy"))
    all_data = data[:,1:]
    all_labels = data[:,0]
    all_labels -= 1

    train_dataset = ManualData(all_data, all_labels, True, multiple_query = multiple_query, size = 28, padding=2, device=device)
    test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, size = 28, padding=2, device=device)
    return train_dataset, test_dataset


def get_purchase_dataset(device='cpu', use_aug = False, multiple_query = False): # 100 categories, 600 dim
    create_path(os.path.join(root, "purchase"))

    data = np.load(os.path.join(root, "purchase", "dataset_purchase.npy"))
    all_data = data[:60000,1:]
    all_labels = data[:60000,0]
    all_labels -= 1


    train_dataset = ManualData(all_data, all_labels, True, multiple_query = multiple_query, size = 28, padding=2, device=device)
    test_dataset = ManualData(all_data, all_labels, use_aug, multiple_query = multiple_query, size = 28, padding=2, device=device)
    return train_dataset, test_dataset

def process_locations_dataset(device='cpu', use_aug = False, multiple_query = False):
    # conver the dataset_locations to npy
    create_path(os.path.join(root, "locations"))

    with open(os.path.join(root, 'locations', 'dataset_locations'), 'r') as file:
        lines = file.readlines()

    print(len(lines))
    for cnt in range(len(lines)):
        line = eval(lines[cnt])
        line = np.array(line, dtype=int)
        if cnt % 1000 == 1:
            print(cnt)
            print(data.shape)
            np.save(os.path.join(root, "locations", "dataset_locations.npy"), data)
        if cnt == 0:
            data = line
            data = data[np.newaxis, :]
        else:
            # concat data and line in dim 0

            data = np.concatenate([data, line[np.newaxis, :]])

    print(data.shape)
    np.save(os.path.join(root, "locations", "dataset_locations.npy"), data)




if __name__ == "__main__":
    # with open("svhn.json") as f:
    #     cfg = json.load(f)
    # train_loader, _ = get_loaders("purchase", "noise", 0, cfg, shuffle=False, batch_size=128, device='cpu', mode = 'train', samplerindex = 0, without_base = True)
    train_dataset, test_dataset = get_svhn_datasets()
    f = open("sampleinfo/target.txt", "r")
    target = eval(f.read())
    f.close()
    train_dataset= Subset(train_dataset, indices = target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    cnt = 0
    print(len(train_loader))
    for x in enumerate(train_loader):
        i, (images, labels) = x
        cnt += images.shape[0]
    print(cnt)