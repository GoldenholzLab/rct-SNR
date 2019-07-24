#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:08
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

num_iter=50

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

srun -c 1 calculate_RMSE.py 