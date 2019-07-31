
monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

min_req_base_sz_count=4
num_baseline_months_per_patient=2
num_testing_months_per_patient=3
num_theo_patients_per_trial_arm=153

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=10
num_iterations=10
num_iter_iter=10

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max

inputs[4]=$min_req_base_sz_count
inputs[5]=$num_baseline_months_per_patient
inputs[6]=$num_testing_months_per_patient
inputs[7]=$num_theo_patients_per_trial_arm

inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma

inputs[12]=$num_trials
inputs[13]=$num_iterations

for ((iter_iter_index=1; iter_iter_index<$num_iter_iter+1; iter_iter_index=iter_iter_index+1))
do

    inputs[14]=$iter_iter_index
    
    bash emp_and_map_based_analysis_wrapper.sh ${inputs[@]}

done