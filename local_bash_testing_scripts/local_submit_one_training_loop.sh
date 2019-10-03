
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

num_trials=10
num_pops=5

training_data_folder_name='training_data_folder'
testing_data_folder_name='testing_data_folder'
RR50_stat_power_model_file_name='RR50_stat_power_model'
num_compute_training_iters=6
num_compute_testing_iters=3

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
inputs[14]=$training_data_folder_name
inputs[15]=$num_compute_training_iters

inputs_two[0]=$monthly_mean_min
inputs_two[1]=$monthly_mean_max
inputs_two[2]=$monthly_std_dev_min
inputs_two[3]=$monthly_std_dev_max
inputs_two[4]=$training_data_folder_name
inputs_two[5]=$RR50_stat_power_model_file_name
inputs_two[6]=$num_compute_training_iters

inputs_three[0]=$monthly_mean_min
inputs_three[1]=$monthly_mean_max
inputs_three[2]=$monthly_std_dev_min
inputs_three[3]=$monthly_std_dev_max
inputs_three[4]=$num_theo_patients_per_trial_arm
inputs_three[5]=$num_baseline_months
inputs_three[6]=$num_testing_months
inputs_three[7]=$minimum_required_baseline_seizure_count
inputs_three[8]=$placebo_mu
inputs_three[9]=$placebo_sigma
inputs_three[10]=$drug_mu
inputs_three[11]=$drug_sigma
inputs_three[12]=$num_trials
inputs_three[13]=$num_pops
inputs_three[14]=$testing_data_folder_name
inputs_three[15]=$num_compute_testing_iters

inputs_four[0]=$monthly_mean_min
inputs_four[1]=$monthly_mean_max
inputs_four[2]=$monthly_std_dev_min
inputs_four[3]=$monthly_std_dev_max
inputs_four[4]=$testing_data_folder_name
inputs_four[5]=$RR50_stat_power_model_file_name
inputs_four[6]=$num_compute_testing_iters

#sbatch submit_generate_data_wrappers.sh ${inputs[@]}
bash local_submit_generate_data_wrappers.sh ${inputs[@]}

all_training_files_exist='False'
while [ "$all_training_files_exist" == "False" ]
do
    sleep 1
    if [ -d $training_data_folder_name ]
    then
        x1=`ls training_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        x2=`ls training_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        x3=`ls training_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        if [ $x1 == $num_compute_training_iters ] && [ $x2 == $num_compute_training_iters ] && [ $x3 == $num_compute_training_iters ]
        then
            all_training_files_exist='True'
            #sbatch train_model_wrapper.sh ${inputs_two[@]}
            bash local_train_model_wrapper.sh ${inputs_two[@]}
        fi
    fi
done

#sbatch submit_generate_data_wrappers.sh ${inputs_three[@]}
bash local_submit_generate_data_wrappers.sh ${inputs_three[@]}

all_testing_files_exist='False'
while [ "$all_testing_files_exist" == "False" ]
do
    sleep 1
    if [ -d $testing_data_folder_name ]
    then
        x1=`ls testing_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        x2=`ls testing_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        x3=`ls testing_data_folder/RR50_emp_stat_powers_*.json | wc -l`
        if [ $x1 == $num_compute_testing_iters ] && [ $x2 == $num_compute_testing_iters ] && [ $x3 == $num_compute_testing_iters ] && [ -f "RR50_stat_power_model_trained.h5" ]
        then
            all_testing_files_exist='True'
            #sbatch test_model_wrapper.sh ${inputs_four[@]}
            bash local_test_model_wrapper.sh ${inputs_four[@]}
        fi
done
