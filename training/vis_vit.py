from models import ResNet18
from models.vit import ViT
from utils import *
import argparse
import os
from torch import optim, nn, einsum
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit')
parser.add_argument('--dataset', type=str, default="CIFAR-10")
parser.add_argument('--load_checkpoint', type=str, default=None)

# General
parser.add_argument('--train_batch', type=int, default=100)
parser.add_argument('--test_batch', type=int, default=1000)

# ViT
parser.add_argument('--dimhead', default="64", type=int)

# Data Augmentation
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")

FLAGS = parser.parse_args()




def main(args):
    if args.dataset=="CIFAR-10":
        image_size = 32
        patch_size = 8
        num_classes = 10
    elif args.dataset=="CIFAR-100":
        image_size = 32
        patch_size = 8
        num_classes = 100
    elif args.dataset=="MNIST" or args.dataset=="FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=int(args.dimhead),
                    depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



if __name__=='__main__':
    main(FLAGS)