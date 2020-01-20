#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

monthly_mean_lower_bound=1
monthly_mean_upper_bound=16
monthly_std_dev_lower_bound=1
monthly_std_dev_upper_bound=16

max_theo_patients_per_trial_arm=300
theo_patients_per_trial_arm_step=5

num_baseline_months=2
num_testing_months=3
baseline_time_scaling_const=1
testing_time_scaling_const=28
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=2000

starting_block=181
num_blocks=50
num_training_files_per_block=15
num_testing_files_per_block=5

data_storage_folder_name='/home/jmr95/rct-SNR/keras_data_and_labels_1-7-2020'

inputs[0]=$monthly_mean_lower_bound
inputs[1]=$monthly_mean_upper_bound
inputs[2]=$monthly_std_dev_lower_bound
inputs[3]=$monthly_std_dev_upper_bound

inputs[4]=$max_theo_patients_per_trial_arm
inputs[5]=$theo_patients_per_trial_arm_step

inputs[6]=$num_baseline_months
inputs[7]=$num_testing_months
inputs[8]=$baseline_time_scaling_const
inputs[9]=$testing_time_scaling_const
inputs[10]=$minimum_required_baseline_seizure_count

inputs[11]=$placebo_mu
inputs[12]=$placebo_sigma
inputs[13]=$drug_mu
inputs[14]=$drug_sigma

inputs[15]=$num_trials

if [ ! -d $data_storage_folder_name ]
then
    mkdir $data_storage_folder_name
fi

for ((block_num=$starting_block; block_num<=$starting_block+$num_blocks-1; block_num=block_num+1))
do

    inputs[16]=$block_num

    inputs[17]="${data_storage_folder_name}/training_data"

    for ((file_index=1; file_index<=$num_training_files_per_block; file_index=file_index+1))
    do
        inputs[18]=$file_index

        sbatch keras_data_generation_wrapper.sh ${inputs[@]}

    done

    inputs[17]="${data_storage_folder_name}/testing_data"

    for ((file_index=1; file_index<=$num_testing_files_per_block; file_index=file_index+1))
    do
        inputs[18]=$file_index

        sbatch keras_data_generation_wrapper.sh ${inputs[@]}

    done

done
