#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:05
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

num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials_per_stat_power_estim=1000
target_stat_power=0.9
num_extra_patients=10

generic_stat_power_model_file_name='stat_power_model_copy'
endpoint_names=("RR50" "MPC" "TTP")
num_iters=3


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
smart_inputs[11]=$num_trials_per_stat_power_estim
smart_inputs[12]=$target_stat_power
smart_inputs[13]=$num_extra_patients
smart_inputs[14]=$generic_stat_power_model_file_name
smart_inputs[15]="smart"

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
dumb_inputs[11]=$num_trials_per_stat_power_estim
dumb_inputs[12]=$target_stat_power
dumb_inputs[13]=$num_extra_patients
dumb_inputs[14]=$generic_stat_power_model_file_name
dumb_inputs[15]="dumb"


for ((iter_index=1; iter_index<=$num_iters; iter_index=iter_index+1))
do
    smart_inputs[16]=$iter_index
    dumb_inputs[16]=$iter_index

    sbatch RR50_cost_wrapper.sh ${smart_inputs[@]}; sbatch RR50_cost_wrapper.sh ${dumb_inputs[@]}
    #sbatch MPC_cost_wrapper.sh  ${smart_inputs[@]}; sbatch MPC_cost_wrapper.sh  ${dumb_inputs[@]}
    #sbatch TTP_cost_wrapper.sh  ${smart_inputs[@]}; sbatch TTP_cost_wrapper.sh  ${dumb_inputs[@]}
done

