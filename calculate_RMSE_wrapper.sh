#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-00:15
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

min_percentile=55
max_percentile=65
num_iter=50
folder='/n/scratch2/jmr95/percentile_question'

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source acativate main_env

srun -c 1 python calculate_RMSE.py $min_percentile $max_percentile $num_iter
