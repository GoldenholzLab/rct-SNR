#!/usr/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

inputs[0]=$1
inputs[1]=$2
inputs[2]=$3
inputs[3]=$4

inputs[4]=$5
inputs[5]=$6
inputs[6]=$7

srun -c 1 python -u main_python_scripts/test_model.py ${inputs[@]}
