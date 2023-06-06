#!/bin/bash

#SBATCH --job-name=llama_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebdaniel@uw.edu,gpsa@uw.edu

#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00  # Max runtime in DD-HH:MM:SS format.
#SBATCH -o ../llama_slurm_outputs/baseline.out

echo "goodluck"

source /gscratch/stf/ebdaniel/miniconda3/bin/activate pytorch

torchrun llama/train_baseline.py
