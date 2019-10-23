#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem-per-cpu=10G
#SBATCH -t 0-02:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

: '
inputs[0]=$1
inputs[1]=$2
inputs[2]=$3
inputs[3]=$4
inputs[4]=$5
inputs[5]=$6
inputs[6]=$7
inputs[7]=$8
inputs[8]=$9
inputs[9]=${10}
inputs[10]=${11}
inputs[11]=${12}
inputs[12]=${13}
inputs[13]=${14}
inputs[14]=${15}
inputs[15]=${16}
inputs[16]=${17}
inputs[17]=${18}
'

monthly_mean_lower_bound=1
monthly_mean_upper_bound=16
monthly_std_dev_lower_bound=1
monthly_std_dev_upper_bound=16

num_theo_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials_per_pop=2000
num_pops=10

data_storage_super_folder_path="/home/jmr95/rct-SNR"
block_generic_folder_name='generic'

block_num=1
compute_iter=1

inputs[0]=$monthly_mean_lower_bound
inputs[1]=$monthly_mean_upper_bound
inputs[2]=$monthly_std_dev_lower_bound
inputs[3]=$monthly_std_dev_upper_bound
inputs[4]=$num_theo_patients_per_trial_arm
inputs[5]=$num_baseline_months
inputs[6]=$num_testing_months
inputs[7]=$minimum_required_baseline_seizure_count
inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma
inputs[12]=$num_trials_per_pop
inputs[13]=$num_pops
inputs[14]=$data_storage_super_folder_path
inputs[15]=$block_generic_folder_name
inputs[16]=$block_num
inputs[17]=$compute_iter

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate working_env

srun -c 1 python -u main_python_scripts/generate_TTP_data.py ${inputs[@]}
