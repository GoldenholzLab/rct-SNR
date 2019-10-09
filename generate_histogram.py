import sys
import os
import keras.models as models
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from main_python_scripts.file_collector import load_iter_specific_files
from main_python_scripts.file_collector import collect_data_from_folder

if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_compute_iters    = 5
    data_storage_folder_name = 'RR50_testing_data_folder_1'
    RR50_stat_power_model_file_name = 'RR50_stat_power_model_1'
    loop_iter = 1

    num_monthly_means    = monthly_mean_max    - (monthly_mean_min    - 1)
    num_monthly_std_devs = monthly_std_dev_max - (monthly_std_dev_min - 1)

    RR50_stat_power_model_file_path = RR50_stat_power_model_file_name + '_' + str(int(loop_iter)) + '.h5'

    [theo_placebo_arm_hists, 
     theo_drug_arm_hists, 
     RR50_emp_stat_powers] = \
         collect_data_from_folder(num_monthly_means,
                                  num_monthly_std_devs,
                                  num_compute_iters,
                                  data_storage_folder_name,
                                  loop_iter)

    RR50_stat_power_model = models.load_model(RR50_stat_power_model_file_name + '_' + str(int(loop_iter)) + '.h5')

    num_inputs = len(RR50_emp_stat_powers)
    error_array = np.zeros(num_inputs)

    for input_index in range(num_inputs):

        prediction = RR50_stat_power_model.predict([theo_placebo_arm_hists[:, :, input_index], theo_drug_arm_hists[:, :, input_index]])
        error = RR50_emp_stat_powers[input_index] - prediction
        error_array[input_index] = error
    
    RMSE = np.sqrt(np.dot(error_array, error_array))

    plt.figure()
    plt.hist(error_array)
    plt.savefig()
    