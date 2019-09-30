#!/usr/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=500M
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

data_storage_folder_name='test_folder'
RR50_stat_power_model_file_name='RR50_stat_power_model.h5'
num_compute_iters=5

inputs2[0]=$monthly_mean_min
inputs2[1]=$monthly_mean_max
inputs2[2]=$monthly_std_dev_min
inputs2[3]=$monthly_std_dev_max

inputs2[4]=$data_storage_folder_name
inputs2[5]=$num_compute_iters
inputs2[6]=$RR50_stat_power_model_file_name

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate working_env

srun -c 1 python -u main_python_scripts/train_model.py ${inputs[@]}