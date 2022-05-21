from models import ResNet18
from models.vit import ViT
from utils import *
import argparse
import os
from torch import optim, nn, einsum
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_idx', default=2)
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--load_checkpoint', type=str, default="../checkpoint/vit_SHVN_e100_b10_lr0.0001.pt")

# General
parser.add_argument('--train_batch', type=int, default=100)
parser.add_argument('--test_batch', type=int, default=1000)

# ViT
parser.add_argument('--dimhead', default=64, type=int)


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

    elif args.dataset == "SVHN":
        image_size = 32
        patch_size = 4
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        norm_trans = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_ds = datasets.SVHN('../data', download=True, transform=transform)

    # Initialize model
    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=args.dimhead,
                    depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    # model = ViT(image_size=32, patch_size=4, num_classes=10, dim=512,
    #             depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    # Load checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_kwargs = {'batch_size': 1, 'shuffle': False}
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
    image_idx = args.image_idx


    # All layers
    for i, (data, target) in enumerate(test_loader):
        if i!=image_idx:
            continue
        image_ori = data[0]
        image = norm_trans(image_ori)

        logits = model(image.unsqueeze(0))
        print(target, logits)
        att_mat = model.get_attn_weights()
        print(f"att_mat has shape {att_mat.shape}")  # [num_layer, num_heads, 17, 17]

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)    # [num_layer, 17, 17]

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # aug_att_mat has shape [num_layer, 17, 17]

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        # joint_attentions has shape [num_layer, 17, 17]

        fig, axs = plt.subplots(2,4)
        image_np = image_ori.permute(1, 2, 0).numpy()
        axs[0,0].imshow(image_np)

        for l in range(joint_attentions.shape[0]):
            # Attention from the output token to the input space.
            v = joint_attentions[l]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = np.array(mask / mask.max())
            # mask = np.array(mask * 10)
            # print(mask)
            c1_mask = cv2.resize(mask, image.size()[1:])[..., np.newaxis]   # (32,32,1)
            mask = np.concatenate((c1_mask, c1_mask, c1_mask), axis=2)      # (32,32,3)

            result = mask * image_np
            idx_0 = int((l+1)/4)
            idx_1 = (l+1)%4

            axs[idx_0,idx_1].set_title(f'L{l+1}')
            _ = axs[idx_0,idx_1].imshow(result)
        plt.show()


if __name__=='__main__':
    main(FLAGS)