#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=cifar-10-resnet
#SBATCH --output=cifar-10-resnet.out

python3 main.py --model="res18" --dataset="CIFAR-10" --epochs=2000 --checkpoint=100
