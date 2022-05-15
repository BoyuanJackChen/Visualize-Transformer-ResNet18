import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(train_loss_list, test_loss_list):
    plt.plot(range(len(train_loss_list)), train_loss_list,
             '-', linewidth=3, label='Train error')
    plt.plot(range(len(test_loss_list)), test_loss_list,
             '-', linewidth=3, label='Test error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def analyze_checkpoint(path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_loss_list = loss[0]
    test_loss_list = loss[1]
    plot_losses(train_loss_list, test_loss_list)
    accuracy_list = loss[2]
    print(np.amax(accuracy_list))


def get_data_loader(args, train_kwargs, test_kwargs):
    if args.dataset == "CIFAR-10":
        # Normalization parameters from https://github.com/kuangliu/pytorch-cifar/issues/19
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261)),
        ])
        dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                                    transform=transform_train)  # 50k
        dataset2 = datasets.CIFAR10('../data', train=False,
                                    transform=transform_test)  # 10k
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    elif args.dataset == "CIFAR-100":
        # Normalization parameters from https://github.com/kuangliu/pytorch-cifar/issues/19
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        dataset1 = datasets.CIFAR100('../data', train=True, download=True,
                                     transform=transform_train)  # 50k
        dataset2 = datasets.CIFAR100('../data', train=False,
                                     transform=transform_test)  # 10k
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    elif args.dataset == "MNIST":
        transf_val = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        if args.transform == "RandomAffine":
            transf_val += [transforms.RandomAffine(
                0, translate=(0.3, 0.5))]
        transform_train = transforms.Compose(transf_val)
        transform_test = transforms.Compose(transf_val)

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform_train)  # 60k
        dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transform_test)  # 10k
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    elif args.dataset == "FashionMNIST":
        transf_val = [
            transforms.ToTensor(),
            transforms.Normalize((0.2859,), (0.3530,)),
        ]
        transform_train = transforms.Compose(transf_val)
        transform_test = transforms.Compose(transf_val)
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform_train)  # 60k
        dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transform_test)  # 10k
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    elif args.dataset == "ImageNet_1k":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        dataset1 = datasets.ImageFolder(
            root='../dataImageNet1K/train', transform=transform)
        dataset2 = datasets.ImageFolder(
            root='../dataImageNet1K/val', transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    path = "../checkpoints/Mixup/lr1e-3/e300_b64_lr0.001.pt"
    analyze_checkpoint(path)
