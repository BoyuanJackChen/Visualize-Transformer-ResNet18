# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer
from randomaug import RandAugment
from torch_cka.cka import CKA



# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
parser.add_argument('--dataset', default='cifar10', type=str, help='Data set')
parser.add_argument('--device', default='cuda:0', type=str, help='Data set')


args = parser.parse_args()


if args.aug:
    import albumentations
bs = int(args.bs)
imsize = int(args.size)

use_amp = args.amp

device = args.device if torch.cuda.is_available() else 'cpu'



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize
transform_train_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_cifar10 = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_train_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test_cifar100 = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_train_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test_svhn = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



if args.dataset == 'cifar10':
    print("cifar10")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar10)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    num_classes_for_training = 10
    
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar100)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar100)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    num_classes_for_training = 100
    
elif args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train_svhn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test_svhn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    num_classes_for_training = 10
    
else:
    print("please use a valid dataset")
    exit()

    


# Model
print('==> Building model..')


net_res = ResNet18(num_classes_for_training)
# ViT for cifar10
net_vit = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes_for_training,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
    
net_res = net_res.to(device)
net_vit = net_vit.to(device)

    
print("args.resume", args.resume)
if args.resume:
    print("resume")
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint1 = torch.load('./checkpoint/res18_'+args.dataset+'-{}-ckpt.t7'.format(args.patch))
    net_res.load_state_dict(checkpoint1['model'])
                             
    checkpoint2 = torch.load('./checkpoint/vit_'+args.dataset+'-{}-ckpt.t7'.format(args.patch))
    net_vit.load_state_dict(checkpoint2['model'])


   

cka = CKA(net_vit, net_res,
        model1_name="ViT", model2_name="ResNet18",
        device=device)

cka.compare(testloader)
cka.plot_results(save_path="/home/haoran/ViT_Small_Dataset/resnet-vit-{}_compare.png".format(args.dataset))