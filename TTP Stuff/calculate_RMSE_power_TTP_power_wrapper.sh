#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-00:10
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

num_stat_power_estimates=1000
bins=100
#folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'
folder='/n/scratch2/jmr95/TTP_stat_power_estimates'

srun -c 1 python -u calculate_RMSE_TTP_power.py $num_stat_power_estimates $bins $folder
#python -u calculate_RMSE_TTP_power.py $num_stat_power_estimates $bins $folder
