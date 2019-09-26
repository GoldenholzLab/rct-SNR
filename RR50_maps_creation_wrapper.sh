

monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

num_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=10

expected_RR50_map_storage_folder=$(pwd)/RR50_test_folder

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_patients_per_trial_arm
inputs[5]=$num_baseline_months
inputs[6]=$num_testing_months
inputs[7]=$minimum_required_baseline_seizure_count
inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma
inputs[12]=$num_trials
inputs[13]=$expected_RR50_map_storage_folder

python -u python_and_R_scripts/RR50_maps_creation.py ${inputs[@]}
