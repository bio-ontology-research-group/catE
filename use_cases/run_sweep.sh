#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J cate
#SBATCH -o out/cate.%J.out
#SBATCH -e err/cate.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]

wandb agent --count 1 ferzcam/cate2/$1
