#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=8:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ImageNet1K
#SBATCH --output=ImageNet1K.out
module purge
singularity exec /scratch/py2097/pytorch.sif /bin/bash -c "source /scratch/py2097/env.sh; python main.py --dataset=ImageNet_1k --epochs=2000 --checkpoint=100 --load_checkpoint=../checkpoint/ImageNet1K_e100_b100_lr0.0001.pt"
