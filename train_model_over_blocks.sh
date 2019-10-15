
python main_python_scripts/initialize_model.py

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

num_epochs=30
num_samples_per_batch=100

training_data_folder_name="/Users/juanromero/Documents/Python_3_Files/rct-SNR_RR50_O2_cluster_data/RR50_training_data"
RR50_stat_power_model_file_name="RR50_stat_power_model"
text_RMSEs_file_name="RMSE_per_block"

num_compute_iters=100
num_train_blocks=19

testing_data_folder_name="/Users/juanromero/Documents/Python_3_Files/rct-SNR_RR50_O2_cluster_data/RR50_training_data"
test_block_num=20
num_bins=100

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_epochs
inputs[5]=$num_samples_per_batch
inputs[6]=$training_data_folder_name
inputs[7]=$RR50_stat_power_model_file_name
inputs[8]=$text_RMSEs_file_name
inputs[9]=$num_compute_iters

inputs_two[0]=$monthly_mean_min
inputs_two[1]=$monthly_mean_max
inputs_two[2]=$monthly_std_dev_min
inputs_two[3]=$monthly_std_dev_max
inputs_two[4]=$testing_data_folder_name
inputs_two[5]=$RR50_stat_power_model_file_name
inputs_two[6]=$num_compute_iters
inputs_two[7]=$test_block_num
inputs_two[8]=$num_bins

for ((train_block_num=0; train_block_num<=$num_train_blocks; train_block_num=train_block_num+1))
do
    echo "training block #: $train_block_num"
    inputs[10]=$train_block_num

    python main_python_scripts/train_model.py ${inputs[@]}
done

python main_python_scripts/generate_histogram.py ${inputs_two[@]}