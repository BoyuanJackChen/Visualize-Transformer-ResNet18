#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=cifar-10
#SBATCH --output=cifar-10.out

python3 main.py --dataset="CIFAR-10" --epochs=2000 --checkpoint=100 --load-checkpoint=None
