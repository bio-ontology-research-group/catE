#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --partition=batch 
#SBATCH --job-name=gocomp
#SBATCH --mail-type=ALL
#SBATCH --output=out/%x-%j-slurm.out
#SBATCH --error=err/%x-%j-slurm.err
#SBATCH --constraint=[a100]


python membership.py --data_root ../../use_cases/ore1/data/ --bs 64 --n_models 5
