
monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

num_theo_patients_per_trial_arm=153
num_baseline_months_per_patient=2
num_testing_months_per_patient=3
min_req_base_sz_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=2
alpha=0.05
folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'
num_stat_power_estimates=1000

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_theo_patients_per_trial_arm
inputs[5]=$num_baseline_months_per_patient
inputs[6]=$num_testing_months_per_patient
inputs[7]=$min_req_base_sz_count
inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma
inputs[12]=$num_trials
inputs[13]=$alpha
inputs[14]=$folder

for ((index=1; index<$num_stat_power_estimates+1; index=index+1))
do
    inputs[15]=$index

    bash cph_power_calc_investigation_wrapper.sh ${inputs[@]}

done
