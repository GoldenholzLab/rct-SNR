
monthly_mean_lower_bound=1
monthly_mean_upper_bound=16
monthly_std_dev_lower_bound=1
monthly_std_dev_upper_bound=16

num_epochs=30
num_samples_per_batch=100

training_data_folder_name="/Users/juanromero/Documents/rct-SNR_O2_generated_data/randomized_theo_pops_10-22-2019/training_data"
testing_data_folder_name="/Users/juanromero/Documents/rct-SNR_O2_generated_data/randomized_theo_pops_10-22-2019/testing_data"
generic_stat_power_model_file_name="stat_power_model"
generic_text_RMSEs_file_name="RMSE_per_block"
endpoint_name="RR50"

num_train_compute_iters=100
num_train_blocks=25

num_test_compute_iters=20
num_test_blocks=25

num_bins=50


inputs[0]=$monthly_mean_lower_bound
inputs[1]=$monthly_mean_upper_bound
inputs[2]=$monthly_std_dev_lower_bound
inputs[3]=$monthly_std_dev_upper_bound
inputs[4]=$generic_stat_power_model_file_name
inputs[5]=$endpoint_name

inputs_two[0]=$monthly_mean_lower_bound
inputs_two[1]=$monthly_mean_upper_bound
inputs_two[2]=$monthly_std_dev_lower_bound
inputs_two[3]=$monthly_std_dev_upper_bound
inputs_two[4]=$num_epochs
inputs_two[5]=$num_samples_per_batch
inputs_two[6]=$training_data_folder_name
inputs_two[7]=$generic_stat_power_model_file_name
inputs_two[8]=$generic_text_RMSEs_file_name
inputs_two[9]=$endpoint_name
inputs_two[10]=$num_train_compute_iters

inputs_three[0]=$monthly_mean_lower_bound
inputs_three[1]=$monthly_mean_upper_bound
inputs_three[2]=$monthly_std_dev_lower_bound
inputs_three[3]=$monthly_std_dev_upper_bound
inputs_three[4]=$testing_data_folder_name
inputs_three[5]=$generic_stat_power_model_file_name
inputs_three[6]=$generic_text_RMSEs_file_name
inputs_three[7]=$endpoint_name
inputs_three[8]=$num_test_compute_iters
inputs_three[9]=$num_test_blocks
inputs_three[10]=$num_bins

python main_python_scripts/initialize_model.py ${inputs[@]}

for ((train_block_num=1; train_block_num<=$num_train_blocks; train_block_num=train_block_num+1))
do
    echo "training on block #$train_block_num"
    inputs_two[11]=$train_block_num

    python main_python_scripts/train_model.py ${inputs_two[@]}
done

python main_python_scripts/generate_histogram.py ${inputs_three[@]}
