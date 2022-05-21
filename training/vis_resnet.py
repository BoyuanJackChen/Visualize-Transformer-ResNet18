import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
from models import ResNet18
import cv2
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_idx', default=2)
parser.add_argument('--dataset', type=str, default="SHVN")
parser.add_argument('--load_checkpoint', type=str, default="../checkpoint/res18_svhn-4-ckpt.t7")

FLAGS = parser.parse_args()


def main(args):
    # Get image
    if args.dataset == "CIFAR-10":
        image_size = 32
        patch_size = 8
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        norm_trans = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        test_ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    elif args.dataset == "CIFAR-100":
        image_size = 32
        patch_size = 8
        num_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        norm_trans = transforms.Compose([
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_ds = datasets.CIFAR100('../data', train=False, download=True, transform=transform)

    elif args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10

    elif args.dataset == "SHVN":
        image_size = 32
        patch_size = 4
        num_class = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        norm_trans = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_ds = datasets.SVHN('../data', download=True, transform=transform)

    test_kwargs = {'batch_size': 1, 'shuffle': False}
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
    image_idx = args.image_idx

    for i, (data, target) in enumerate(test_loader):
        if i!=image_idx:
            continue
        image_ori = data[0]
        image = norm_trans(image_ori)

    image_np = np.asarray(image_ori.permute(1,2,0))

    # Prepare the model
    # model = models.resnet18(pretrained=True)
    model = ResNet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load("../checkpoint/res18_CIFAR-10_e200_b100_lr0.0001.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model_weights =[]
    conv_layers = []
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # Append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    fig = plt.figure(figsize=(90, 50))
    a = fig.add_subplot(3, 6, 1)
    image_gray = rgb2gray(image_np)
    image_gray = cv2.resize(image_gray, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    imgplot = plt.imshow(image_gray, cmap="gray")
    a.axis("off")
    for i in range(len(processed)):
        a = fig.add_subplot(3, 6, i+2)
        imgplot = plt.imshow(processed[i], cmap="gray")
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')



if __name__=="__main__":
    main(FLAGS)
