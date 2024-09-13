### 1. Dependencies
- CUDA 11.0
- You can install the required packages by running: ```pip install -r requirements.txt```

### 2. Datasets
- CIFAR-10, CIFAR-100 can be downloaded directly from torchvision.datasets.
- Purchase and Locations can be downloaded from https://github.com/privacytrustlab/datasets


### 4. Usage

This repository contains the code for training shadow models and performing LiRA. We support 12 data enhancement methods: "base", "smooth", "disturblabel", "noise", "cutout", "mixup", "jitter", "pgdat", "trades", "distillation", "AWP", "TradesAWP". The following steps are the instructions for reproducing the results in the paper. On CIFAR-10, we take one data augmentation method, Cutout, as an example:

Train the 128 shadow models for Cutout:
```
python train.py --train --s_model 0 --t_model 128 --aug_type cutout --dataset cifar10
```


Compute $\phi$ (required by LiRA) for each data point with multiple queries (Ensuring all 128 models trained):
```
python inference.py --mode eval --load_model --save_results --dataset cifar10 --query_mode multiple --aug_type cutout
```

Perform LiRA attacck after all $\phi$ computed (multiple queries version):
```
python eval_privacy.py --save_results --multi
```


### 5. Statement

