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

max_theo_patients_per_trial_arm=300
theo_patients_per_trial_arm_step=50

num_baseline_months=2
num_testing_months=3
baseline_time_scaling_const=1
testing_time_scaling_const=28
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=20

folder='/home/jmr95/rct-SNR/test_algo'

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max

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

if [ ! -f $folder ]
then
    mkdir $folder
fi

inputs[16]="${folder}/training_data"

for ((file_index=1; file_index<=10; file_index=file_index+1))
do
    inputs[17]=$file_index

    sbatch keras_data_generation_wrapper.sh ${inputs[@]}

done

inputs[16]="${folder}/testing_data"

for ((file_index=1; file_index<=2; file_index=file_index+1))
do
    inputs[17]=$file_index

    sbatch keras_data_generation_wrapper.sh ${inputs[@]}

done

