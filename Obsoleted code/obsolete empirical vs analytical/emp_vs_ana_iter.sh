
monthly_mean_min=4
monthly_mean_max=6
monthly_std_dev_min=1
monthly_std_dev_max=8

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_theo_patients_per_trial_arm=153
num_baseline_months_per_patient=2
num_testing_months_per_patient=3
min_req_bs_sz_count=4

num_trials=10
num_iter=3

min_percentile=55
max_percentile=60

inputs[1]=$monthly_mean_min
inputs[2]=$monthly_mean_max
inputs[3]=$monthly_std_dev_min
inputs[4]=$monthly_std_dev_max
inputs[5]=$placebo_mu
inputs[6]=$placebo_sigma
inputs[7]=$drug_mu
inputs[8]=$drug_sigma
inputs[9]=$num_theo_patients_per_trial_arm
inputs[10]=$num_baseline_months_per_patient
inputs[11]=$num_testing_months_per_patient
inputs[12]=$min_req_bs_sz_count
inputs[13]=$num_trials
inputs[14]='/users/juanromero/Documents/Python_3_Files/useless_folder'

for ((percentile=min_percentile; percentile<max_percentile+1; percentile=percentile+1))
do
    inputs[15]=$percentile
    for ((iter=1; iter<num_iter+1; iter=iter+1))
    do
        inputs[16]=$iter
        bash test_empirical_vs_analytical.sh ${inputs[@]}
    done
done