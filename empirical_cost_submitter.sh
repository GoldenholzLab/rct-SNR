
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
O2_attempt_num=2

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_baseline_months
inputs[5]=$num_testing_months
inputs[6]=$minimum_required_baseline_seizure_count
inputs[7]=$placebo_mu
inputs[8]=$placebo_sigma
inputs[9]=$drug_mu
inputs[10]=$drug_sigma
inputs[11]=$target_stat_power
inputs[12]=$num_trials
inputs[13]=$SNR_num_extra_patients_per_trial_arm
inputs[14]=$O2_attempt_num

for ((iter_index=1; iter_index<=$num_iters; iter_index=iter_index+1))
do

    inputs[15]=$iter_index
    inputs[15]=$iter_index

    sbatch RR50_empirical_cost_wrapper.sh ${dumb_inputs[@]}
    sbatch RR50_empirical_cost_wrapper.sh ${smart_inputs[@]}
done
