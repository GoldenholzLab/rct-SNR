
start_monthly_mean=0
stop_monthly_mean=4
step_monthly_mean=1
start_monthly_std_dev=0
stop_monthly_std_dev=5
step_monthly_std_dev=1
min_req_base_sz_count=0
num_patients_per_trial_arm=153
num_months_baseline=2
num_months_testing=3
placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05
num_trials=3

inputs[0]=$start_monthly_mean
inputs[1]=$stop_monthly_mean
inputs[2]=$step_monthly_mean
inputs[3]=$start_monthly_std_dev
inputs[4]=$stop_monthly_std_dev
inputs[5]=$step_monthly_std_dev
inputs[6]=$min_req_base_sz_count
inputs[7]=$num_patients_per_trial_arm
inputs[8]=$num_months_baseline
inputs[9]=$num_months_testing
inputs[10]=$placebo_mu
inputs[11]=$placebo_sigma
inputs[12]=$drug_mu
inputs[13]=$drug_sigma
inputs[14]=$num_trials

python estimate_voxel_runtimes.py ${inputs[@]}
