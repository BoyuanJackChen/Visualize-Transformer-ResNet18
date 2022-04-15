from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--CIFAR-10', type=bool, default=True)
parser.add_argument('--MNIST', type=bool, default=True)
parser.add_argument('--FashionMNIST', type=bool, default=True)
FLAGS = parser.parse_args()

def main(args):
    print(f"Your torch version is {torch.__version__}")
    train_kwargs = {'batch_size': 10}
    test_kwargs = {'batch_size': 10}

    if args.CIFAR_10:
        train_ds = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 50k
        test_ds = datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
        # show_dataset_info(train_ds, test_ds)

    if args.MNIST:
        train_ds = datasets.MNIST(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 60k
        test_ds = datasets.MNIST(
            root='../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        # show_dataset_info(train_ds, test_ds)

    if args.FashionMNIST:
        train_ds = datasets.FashionMNIST(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 60k
        test_ds = datasets.FashionMNIST(
            root='../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        show_dataset_info(train_ds, test_ds)

def show_dataset_info(train_ds, test_ds):
    print(f"Train dataset has length {len(train_ds)}")
    print(f"Test dataset has length {len(test_ds)}")
    print("Train set label counts:", torch.IntTensor(train_ds.targets).bincount())
    print("Test set label counts:",  torch.IntTensor(test_ds.targets).bincount())
    sample = next(iter(train_ds))
    image, label = sample
    print("Data shape is", image.shape)
    plt.imshow(image.squeeze().T)
    plt.show()

if __name__=="__main__":
    main(FLAGS)