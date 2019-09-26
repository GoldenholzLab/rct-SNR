
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

num_trials=500
num_patients_per_map_location=50
alpha=0.05

folder=$(pwd)/test_folder_2
RMSE_folder=$(pwd)/RMSE_test_folder
num_iter=10

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
inputs[13]=$num_patients_per_map_location
inputs[14]=$alpha
inputs[15]=$folder
inputs[16]=$RMSE_folder

for ((estim_iter=1; estim_iter<=num_iter; estim_iter=estim_iter+1))
do
    inputs[17]=$estim_iter

    bash TTP_emp_and_map_power_estimation_wrapper.sh ${inputs[@]}
done

