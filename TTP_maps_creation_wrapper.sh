#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-01:30
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

monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

num_patients_per_map_location=50
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05
folder=$(pwd)/test_folder_2

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_patients_per_map_location
inputs[5]=$num_baseline_months
inputs[6]=$num_testing_months
inputs[7]=$minimum_required_baseline_seizure_count
inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma
inputs[12]=$folder

srun -c 1 python -u python_and_R_scripts/TTP_maps_creation.py ${inputs[@]}
