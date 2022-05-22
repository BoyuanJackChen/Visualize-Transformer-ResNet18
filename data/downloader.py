from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--CIFAR-10', type=bool, default=True)
parser.add_argument('--CIFAR-100', type=bool, default=True)
parser.add_argument('--SVHN', type=bool, default=True)
parser.add_argument('--MNIST', type=bool, default=False)
parser.add_argument('--FashionMNIST', type=bool, default=False)
parser.add_argument('--view-sample', type=bool, default=False)
FLAGS = parser.parse_args()

def main(args):
    # Use gpu if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on device: {torch.cuda.get_device_name(0)}")

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
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        if args.view_sample:
            print()
            print("CIFAR-10:")
            show_dataset_info(train_ds, test_ds)
    
    if args.SVHN:
        ds = datasets.SVHN(
            root='../data',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))

    if args.CIFAR_100:
        train_ds = datasets.CIFAR100(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 50k
        test_ds = datasets.CIFAR100(
            root='../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        if args.view_sample:
            print()
            print("CIFAR-100:")
            show_dataset_info(train_ds, test_ds)

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
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        if args.view_sample:
            print()
            print("MNIST:")
            show_dataset_info(train_ds, test_ds)

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
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))  # 10k
        if args.view_sample:
            print()
            print("FashionMNIST:")
            show_dataset_info(train_ds, test_ds)

def show_dataset_info(train_ds, test_ds):
    print(f"Train dataset has length {len(train_ds)}")
    print(f"Test dataset has length {len(test_ds)}")
    print("Train set label counts:", np.bincount(np.array(train_ds.targets)))
    print("Test set label counts:",  np.bincount(np.array(test_ds.targets)))
    sample = next(iter(train_ds))
    image, label = sample
    print("Data shape is", image.shape)

def view_sample(train_ds):
    while True:
        sample = next(iter(train_ds))
        image, label = sample
        plt.imshow(image.squeeze().T)
        plt.show()

if __name__=="__main__":
    main(FLAGS)