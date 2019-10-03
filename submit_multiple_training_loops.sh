#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=500M
#SBATCH -t 0-00:10
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

num_theo_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=10
num_pops=5

training_data_folder_name='training_data_folder'
testing_data_folder_name='testing_data_folder'
RR50_stat_power_model_file_name='RR50_stat_power_model'
num_compute_training_iters=6
num_compute_testing_iters=3


inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$RR50_stat_power_model_file_name

inputs_two[0]=$monthly_mean_min
inputs_two[1]=$monthly_mean_max
inputs_two[2]=$monthly_std_dev_min
inputs_two[3]=$monthly_std_dev_max
inputs_two[4]=$num_theo_patients_per_trial_arm
inputs_two[5]=$num_baseline_months
inputs_two[6]=$num_testing_months
inputs_two[7]=$minimum_required_baseline_seizure_count
inputs_two[8]=$placebo_mu
inputs_two[9]=$placebo_sigma
inputs_two[10]=$drug_mu
inputs_two[11]=$drug_sigma
inputs_two[12]=$num_trials
inputs_two[13]=$num_pops
inputs_two[14]=$training_data_folder_name
inputs_two[15]=$testing_data_folder_name
inputs_two[16]=$RR50_stat_power_model_file_name
inputs_two[17]=$num_compute_training_iters
inputs_two[18]=$num_compute_testing_iters

sbatch initialize_model_wrapper.sh ${inputs[@]}

while [ ! -f "$RR50_stat_power_model_file_name.h5" ]
do
    sleep 1
done

sbatch submit_one_training_loop.sh ${inputs_two[@]}
