from models import ResNet18
from models.vit import ViT
from utils import *
import argparse
import os
from torch import optim, nn, einsum
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit')
parser.add_argument('--dataset', type=str, default="CIFAR-10")
parser.add_argument('--load_checkpoint', type=str, default="../checkpoint/CIFAR-10_e100_b100_lr0.0001.pt")

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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
    elif args.dataset=="CIFAR-100":
        image_size = 32
        patch_size = 8
        num_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif args.dataset=="MNIST" or args.dataset=="FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10

    # Initialize model
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=int(args.dimhead),
                    depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    # Load checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_kwargs = {'batch_size': 1, 'shuffle': False}
    test_ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
    for i, (data, target) in enumerate(test_loader):
        if i!=10:
            continue
        image = data[0]
        print(image.shape)
        print(target)
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

        logits = model(image.unsqueeze(0))
        attn_map = model.get_attn_weights()
        print(attn_map.shape)
        # print(att_mat.shape)
        # # Average the attention weights across all heads.
        # att_mat = torch.mean(att_mat, dim=1)



if __name__=='__main__':
    main(FLAGS)