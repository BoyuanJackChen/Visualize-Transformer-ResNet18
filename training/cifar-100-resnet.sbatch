#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=cifar-100-resnet
#SBATCH --output=cifar-100-resnet.out

python3 main.py --model="res18" --dataset="CIFAR-100" --epochs=2000 --checkpoint=100
