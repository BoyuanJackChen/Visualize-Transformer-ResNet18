from model import *
from utils import *
import time
from torch import optim
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--checkpoint', type=int, default=20)
parser.add_argument('--load_checkpoint', type=str, default=None)
parser.add_argument('--dataset', type=str, default="FashionMNIST")
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--train_batch', type=int, default=100)
parser.add_argument('--test_batch', type=int, default=1000)
FLAGS = parser.parse_args()


def main(args):
    # Use gpu if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
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

    train_loader, test_loader = get_data_loader(args, train_kwargs, test_kwargs)
    if args.dataset=="CIFAR-10":
        image_size = 32
        patch_size = 8
        num_classes = 10
    elif args.dataset=="MNIST" or args.dataset=="FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=10, channels=1,
                dim=64, depth=4, heads=8, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model has {count_parameters(model)} parameters")

    # Load Checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        train_loss_list = loss[0]
        test_loss_list = loss[1]
        accuracy_list = loss[2]
        start_epoch = checkpoint['epoch']+1

    # Training
    start_time = time.time()
    train_loss_history, test_loss_history = [], []
    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)



def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('['+'{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')



if __name__=="__main__":
    main(FLAGS)