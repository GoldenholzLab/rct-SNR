
monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

data_storage_folder_name=test_folder
num_compute_iters=5
RR50_stat_power_model_file_name='RR50_stat_power_model.h5'

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max

inputs[4]=$data_storage_folder_name
inputs[5]=$num_compute_iters
inputs[6]=$RR50_stat_power_model_file_name

python -u main_python_scripts/train_model.py ${inputs[@]}
