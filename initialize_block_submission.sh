
monthly_mean_lower_bound=1
monthly_mean_upper_bound=16
monthly_std_dev_lower_bound=1
monthly_std_dev_upper_bound=16

max_theo_patients_per_trial_arm=300
theo_patients_per_trial_arm_step=100

num_baseline_months=2
num_testing_months=3
baseline_time_scaling_const=1
testing_time_scaling_const=28
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=5

num_training_files_per_block=2
num_testing_files_per_block=2

data_storage_folder_name='/home/jmr95/rct-SNR/keras_data_and_labels_1-7-2020'

if [ ! -d $data_storage_folder_name ]
then
    mkdir $data_storage_folder_name
fi

inputs[0]=$monthly_mean_lower_bound
inputs[1]=$monthly_mean_upper_bound
inputs[2]=$monthly_std_dev_lower_bound
inputs[3]=$monthly_std_dev_upper_bound

inputs[4]=$max_theo_patients_per_trial_arm
inputs[5]=$theo_patients_per_trial_arm_step

inputs[6]=$num_baseline_months
inputs[7]=$num_testing_months
inputs[8]=$baseline_time_scaling_const
inputs[9]=$testing_time_scaling_const
inputs[10]=$minimum_required_baseline_seizure_count

inputs[11]=$placebo_mu
inputs[12]=$placebo_sigma
inputs[13]=$drug_mu
inputs[14]=$drug_sigma

inputs[15]=$num_trials
inputs[16]=$data_storage_folder_name
inputs[17]=1

inputs[18]=$num_training_files_per_block
inputs[19]=$num_testing_files_per_block

bash submit_one_block.sh ${inputs[@]}

