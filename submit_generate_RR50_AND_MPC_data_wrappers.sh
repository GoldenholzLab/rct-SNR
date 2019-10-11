#!/usr/bin/bash

#SBATCH -p short
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

num_theo_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=2000
num_pops=10

data_storage_super_folder_path="/home/jmr95/rct-SNR"
RR50_training_loop_iter_specific_file_name="RR50_training_data"
RR50_testing_loop_iter_specific_file_name="RR50_testing_data"

num_training_compute_iters_per_loop=100
num_testing_compute_iters_per_loop=20
num_loop_iters=25


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
inputs[14]=$data_storage_super_folder_path
inputs[15]=$RR50_training_loop_iter_specific_file_name

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
inputs_two[14]=$data_storage_super_folder_path
inputs_two[15]=$RR50_testing_loop_iter_specific_file_name


for ((loop_iter=0; loop_iter<$num_loop_iters; loop_iter=loop_iter+1))
do
    for ((compute_iter=1; compute_iter<$num_training_compute_iters_per_loop+1; compute_iter=compute_iter+1))
    do
        inputs[16]=$loop_iter
        inputs[17]=$compute_iter

        #bash generate_RR50_and_MPC_data_wrapper.sh ${inputs[@]}
        sbatch generate_RR50_and_MPC_data_wrapper.sh ${inputs[@]}
    done
done

for ((loop_iter=0; loop_iter<$num_loop_iters; loop_iter=loop_iter+1))
do
    for ((compute_iter=1; compute_iter<$num_testing_compute_iters_per_loop+1; compute_iter=compute_iter+1))
    do
        inputs_two[16]=$loop_iter
        inputs_two[17]=$compute_iter

        #bash generate_RR50_and_MPC_data_wrapper.sh ${inputs_two[@]}
        sbatch generate_RR50_and_MPC_data_wrapper.sh ${inputs_two[@]}
    done
done
