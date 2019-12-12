#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

# SNR map parameters
monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

# RCT design parameters
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

# simulation parameters
placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05
target_stat_power=0.9

# computational estimation parameters
num_trials=2000
SNR_num_extra_patients_per_trial_arm=20

# parallel processing parameters
num_iters=200

dumb_inputs[0]=$monthly_mean_min
dumb_inputs[1]=$monthly_mean_max
dumb_inputs[2]=$monthly_std_dev_min
dumb_inputs[3]=$monthly_std_dev_max
dumb_inputs[4]=$num_baseline_months
dumb_inputs[5]=$num_testing_months
dumb_inputs[6]=$minimum_required_baseline_seizure_count
dumb_inputs[7]=$placebo_mu
dumb_inputs[8]=$placebo_sigma
dumb_inputs[9]=$drug_mu
dumb_inputs[10]=$drug_sigma
dumb_inputs[11]=$target_stat_power
dumb_inputs[12]=$num_trials
dumb_inputs[13]=$SNR_num_extra_patients_per_trial_arm
dumb_inputs[14]="dumb"

smart_inputs[0]=$monthly_mean_min
smart_inputs[1]=$monthly_mean_max
smart_inputs[2]=$monthly_std_dev_min
smart_inputs[3]=$monthly_std_dev_max
smart_inputs[4]=$num_baseline_months
smart_inputs[5]=$num_testing_months
smart_inputs[6]=$minimum_required_baseline_seizure_count
smart_inputs[7]=$placebo_mu
smart_inputs[8]=$placebo_sigma
smart_inputs[9]=$drug_mu
smart_inputs[10]=$drug_sigma
smart_inputs[11]=$target_stat_power
smart_inputs[12]=$num_trials
smart_inputs[13]=$SNR_num_extra_patients_per_trial_arm
smart_inputs[14]="smart"

for ((iter_index=1; iter_index<=$num_iters; iter_index=iter_index+1))
do

    echo $iter_index

    dumb_inputs[15]=$iter_index
    smart_inputs[15]=$iter_index

    sbatch RR50_empirical_cost_wrapper.sh ${dumb_inputs[@]}; sbatch RR50_empirical_cost_wrapper.sh ${smart_inputs[@]}
done
