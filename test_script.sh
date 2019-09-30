#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

inputs2[0]=$monthly_mean_min
inputs2[1]=$monthly_mean_max
inputs2[2]=$monthly_std_dev_min
inputs2[3]=$monthly_std_dev_max

inputs2[4]=$data_storage_folder_name
inputs2[5]=$RR50_stat_power_model_file_name
inputs2[6]=$num_compute_iters

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

srun -c 1 main_python_scripts/data_watcher.py ${inputs2[@]}