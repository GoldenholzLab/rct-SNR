#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-00:45
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o jmr95_%j.out
#SBATCH -e jmr95_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
module load R/3.5.1

source activate main_env

srun -c 1 python random_population.py $1