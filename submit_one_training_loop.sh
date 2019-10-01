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

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

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
data_storage_folder_name='test_folder'
RR50_stat_power_model_file_name='RR50_stat_power_model.h5'
num_compute_iters=5

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max

inputs[4]=$num_theo_patients_per_trial_arm
inputs[5]=$num_baseline_months
inputs[6]=$num_testing_months
inputs[7]=$minimum_required_baseline_seizure_count

inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma

inputs[12]=$num_trials
inputs[13]=$num_pops
inputs[14]=$data_storage_folder_name

inputs_two[0]=$monthly_mean_min
inputs_two[1]=$monthly_mean_max
inputs_two[2]=$monthly_std_dev_min
inputs_two[3]=$monthly_std_dev_max

inputs_two[4]=$data_storage_folder_name
inputs_two[5]=$RR50_stat_power_model_file_name
inputs_two[6]=$num_compute_iters

for ((compute_iter=1; compute_iter<num_compute_iters+1; compute_iter=compute_iter+1))
do
    inputs[15]=$compute_iter

    sbatch generate_data_wrapper.sh ${inputs[@]}

done

sbatch data_watcher_wrapper.sh ${inputs_two[@]}
