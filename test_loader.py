import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np
import os
from torchvision import datasets, transforms

root = "/Users/luo/reproduce/privacy_and_aug"
train_ds = datasets.CIFAR100(os.path.join(root, "cifar100"), train=True, download=True)


# 加载保存的索引
loaded_mem_equal_0 = np.load('sampleinfo/mem_equal_0.npy')
loaded_mem_equal_1 = np.load('sampleinfo/mem_equal_1.npy')

# 创建自定义的 Sampler
class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# 创建使用自定义顺序的 Subset 和 DataLoader
train_subset_0 = Subset(train_ds, loaded_mem_equal_0)
train_loader_0 = DataLoader(train_subset_0, batch_size=64, shuffle=False, num_workers=0, sampler=CustomSampler(loaded_mem_equal_0))

train_subset_1 = Subset(train_ds, loaded_mem_equal_1)
train_loader_1 = DataLoader(train_subset_1, batch_size=64, shuffle=False, num_workers=0, sampler=CustomSampler(loaded_mem_equal_1))

# 打印示例批次以验证加载顺序
for batch in train_loader_0:
    print(batch)
    break

for batch in train_loader_1:
    print(batch)
    break