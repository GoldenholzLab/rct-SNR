
monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

num_theo_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=100
num_pops=5

data_storage_super_folder_path="/Users/juanromero/Documents/Python_3_Files/rct_SNR_test"
loop_iter_specific_file_name="RR50_training_data"

num_loop_iters=3
num_compute_iters_per_loop=3

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
inputs[13]=$num_pops

inputs[14]=$data_storage_super_folder_path
inputs[15]=$loop_iter_specific_file_name


for ((loop_iter=0; loop_iter<$num_loop_iters; loop_iter=loop_iter+1))
do

    for ((compute_iter=1; compute_iter<$num_compute_iters_per_loop+1; compute_iter=compute_iter+1))
    do
    
        inputs[16]=$loop_iter
        inputs[17]=$compute_iter

        bash generate_data_wrapper.sh ${inputs[@]}

    done

done

