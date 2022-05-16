from models import ResNet18
from utils import *
import time
import argparse
import os
from torch import optim, nn, einsum
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit')
parser.add_argument('--dataset', type=str, default="CIFAR-10")
# --transform=RandomAffine for random translate
parser.add_argument('--transform', type=str, default="None")
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--checkpoint', type=int, default=100)
parser.add_argument('--load_checkpoint', type=str,
                    default="../checkpoint/CIFAR-10_e100_b100_lr0.0001.pt")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

# General
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--train_batch', type=int, default=100)
parser.add_argument('--test_batch', type=int, default=1000)

# ViT
parser.add_argument('--dimhead', default="64", type=int)

# CNN
parser.add_argument('--convkernel', default='8', type=int)

# Data Augmentation
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true',
                    help='add mixup augumentations')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")

FLAGS = parser.parse_args()


def main(args):
    # Use gpu if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device is:", device)
    print(f"Running on device: {torch.cuda.get_device_name(0)}")
    # Parameters
    train_kwargs = {'batch_size': args.train_batch, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Checkpoint saving and loading
    PATH = "../checkpoint/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Function from utils. Normalization is implemented
    train_loader, test_loader = get_data_loader(
        args, train_kwargs, test_kwargs)

    print('==> Loading Dataset..')
    # patch_size is the number of pixels for each patch's width and height. Not patch number.
    if args.dataset == "CIFAR-10":
        image_size = 32
        patch_size = 8
        num_classes = 10
    elif args.dataset == "CIFAR-100":
        image_size = 32
        patch_size = 8
        num_classes = 100
    elif args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10
    elif args.dataset == "ImageNet_1k":
        image_size = 224
        patch_size = 56
        num_classes = 1000

    print('==> Building model..')
    if args.model == 'res18':
        model = ResNet18()
    elif args.model == "vit":
        from models.vit import ViT
        model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=int(args.dimhead),
                    depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    elif args.model == "vit_small":
        from models.vit_small import ViT
        model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=int(args.dimhead),
                    depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    print(f"Model has {count_parameters(model)} parameters")
    if device == 'cuda':
        model = torch.nn.DataParallel(model)    # make parallel
        cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    start_epoch = 1
    train_loss_history, test_loss_history, test_accuracy_history = np.array(
        []), np.array([]), np.array([])
    # remove this condition when making a new checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        train_loss_history = checkpoint['train_loss']
        test_loss_history = checkpoint['test_loss']
        test_accuracy_history = checkpoint['accuracy']
    print('==> Training starts')
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        print('Epoch:', epoch)
        train_epoch(model, device, optimizer, criterion,
                    train_loader, train_loss_history)
        evaluate_model(model, device, test_loader, test_loss_history,
                       test_accuracy_history)
        print('Epoch took:', '{:5.2f}'.format(
            time.time() - start_time), 'seconds')

        if epoch % args.checkpoint == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_history,
                'test_loss': test_loss_history,
                'accuracy': test_accuracy_history,
            }, PATH + f"/{args.dataset}_e{epoch}_b{args.train_batch}_lr{args.lr}.pt")
            print(f"Checkpoint {args.dataset}_e{epoch}_b{args.train_batch}_lr{args.lr}.pt saved")


def train_epoch(model, device, optimizer, criterion, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('['+'{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history = np.append(loss_history, loss.item())


def evaluate_model(model, device, data_loader, loss_history, accuracy_history):
    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()

    avg_loss = total_loss / total_samples
    loss_history = np.append(loss_history, avg_loss)
    accuracy = correct_samples / total_samples
    accuracy_history = np.append(accuracy_history, accuracy)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


if __name__ == "__main__":
    main(FLAGS)
