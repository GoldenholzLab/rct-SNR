
monthly_mean_lower_bound=1
monthly_mean_upper_bound=16
monthly_std_dev_lower_bound=1
monthly_std_dev_upper_bound=16

num_epochs=30
num_samples_per_batch=100

training_data_folder_name="/Users/juanromero/Documents/rct-SNR_o2_generated_data/keras_data_and_labels_11-14-2019/training_data"
testing_data_folder_name="/Users/juanromero/Documents/rct-SNR_o2_generated_data/keras_data_and_labels_11-14-2019/testing_data"
generic_stat_power_model_file_name="stat_power_model"
generic_text_RMSEs_file_name="RMSE_per_block"
model_errors_file_name="model_errors"

RR50_endpoint_name="RR50"
MPC_endpoint_name="MPC"
TTP_endpoint_name="TTP"

num_blocks=25
num_train_compute_iters_per_block=15
num_test_compute_iters_per_block=5

error_histogram_bins=100

num_theo_patients_per_trial_arm_in_snr_map=50
num_theo_patients_per_trial_arm_in_snr_map_loc=20
num_hists_per_trial_arm=500

max_SNR=30
min_SNR=-20

endpoint_names[0]=$RR50_endpoint_name
endpoint_names[1]=$MPC_endpoint_name
endpoint_names[2]=$TTP_endpoint_name

inputs[0]=$monthly_mean_lower_bound
inputs[1]=$monthly_mean_upper_bound
inputs[2]=$monthly_std_dev_lower_bound
inputs[3]=$monthly_std_dev_upper_bound
inputs[4]=$generic_stat_power_model_file_name

inputs_two[0]=$monthly_mean_lower_bound
inputs_two[1]=$monthly_mean_upper_bound
inputs_two[2]=$monthly_std_dev_lower_bound
inputs_two[3]=$monthly_std_dev_upper_bound
inputs_two[4]=$num_epochs
inputs_two[5]=$num_samples_per_batch
inputs_two[6]=$training_data_folder_name
inputs_two[7]=$generic_stat_power_model_file_name
inputs_two[8]=$generic_text_RMSEs_file_name
inputs_two[9]=$num_train_compute_iters_per_block

inputs_three[0]=$monthly_mean_lower_bound
inputs_three[1]=$monthly_mean_upper_bound
inputs_three[2]=$monthly_std_dev_lower_bound
inputs_three[3]=$monthly_std_dev_upper_bound
inputs_three[4]=$testing_data_folder_name
inputs_three[5]=$generic_stat_power_model_file_name
inputs_three[6]=$generic_text_RMSEs_file_name
inputs_three[7]=$model_errors_file_name
inputs_three[8]=$num_test_compute_iters_per_block
inputs_three[9]=$num_blocks

inputs_four[0]=$num_theo_patients_per_trial_arm_in_snr_map
inputs_four[1]=$num_theo_patients_per_trial_arm_in_snr_map_loc
inputs_four[2]=$num_hists_per_trial_arm

inputs_five[0]=$model_errors_file_name
inputs_five[1]=$error_histogram_bins

inputs_six[0]=$max_SNR
inputs_six[1]=$min_SNR

for ((endpoint_name_index=0; endpoint_name_index<=2; endpoint_name_index=endpoint_name_index+1))
do

    endpoint_name=${endpoint_names[$endpoint_name_index]}

    inputs[5]=$endpoint_name
    inputs_two[10]=$endpoint_name
    inputs_three[10]=$endpoint_name
    inputs_four[3]=$endpoint_name

    python main_python_scripts/initialize_model.py ${inputs[@]}

    for ((block_num=1; block_num<=$num_blocks; block_num=block_num+1))
    do
        echo "training on block #$block_num"
        inputs_two[11]=$block_num

        python main_python_scripts/train_model.py ${inputs_two[@]}
    done

    python main_python_scripts/test_model.py ${inputs_three[@]}

    python main_python_scripts/generate_SNR_data.py ${inputs_four[@]}

done

python main_python_scripts/plot_histograms.py ${inputs_five[@]}

python main_python_scripts/plot_SNR_heatmap.py ${inputs_six[@]}
