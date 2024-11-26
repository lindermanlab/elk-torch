#!/usr/bin/bash
#SBATCH --job-name=seq_mnist_torch
#SBATCH --error=seq_mnist_torch_%j_%a.err
#SBATCH --out=seq_mnist_torch_%j_%a.out
#SBATCH --time=02:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL

module load cuda/12.4


source /scratch/users/xavier18/miniconda3/bin/activate elk_torch

# Debugging
env | grep PATH
which ninja
python -m pip show ninja


/scratch/users/xavier18/miniconda3/envs/elk_torch/bin/python3 seq_mnist_torch.py $@
