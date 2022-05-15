#!/bin/bash 
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=6:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=cifar-100
#SBATCH --output=cifar-100.out

python3 main.py --dataset="CIFAR-100" --epochs=2000 --checkpoint=100
