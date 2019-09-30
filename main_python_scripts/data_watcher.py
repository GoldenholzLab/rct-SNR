import os
import numpy as np
from time import sleep
import subprocess
import sys


if(__name__=='__main__'):

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    data_storage_folder_name        = sys.argv[5]
    RR50_stat_power_model_file_name = sys.argv[6]
    num_compute_iters           = int(sys.argv[7])
        
    #num_minutes_to_wait = 5
    num_seconds_to_wait = 5

    data_storage_folder_file_path = './' + data_storage_folder_name
    all_iter_files_exist = False
    #num_seconds_to_wait = num_minutes_to_wait*60

    while(not all_iter_files_exist):

        each_iter_exists = np.zeros(num_compute_iters, dtype=bool)

        for compute_iter_index in range(num_compute_iters):

            iter_specific_theo_placebo_arm_hists_file_name = 'theo_placebo_arm_hists_' + str(compute_iter_index + 1) + '.json'
            iter_specific_theo_drug_arm_hists_file_name    = 'theo_drug_arm_hists_'    + str(compute_iter_index + 1) + '.json'
            iter_specific_RR50_emp_stat_powers_file_name   = 'RR50_emp_stat_powers_'   + str(compute_iter_index + 1) + '.json'

            iter_specific_theo_placebo_arm_hists_file_path = data_storage_folder_file_path + '/' + iter_specific_theo_placebo_arm_hists_file_name
            iter_specific_theo_drug_arm_hists_file_path    = data_storage_folder_file_path + '/' + iter_specific_theo_drug_arm_hists_file_name
            iter_specific_RR50_emp_stat_powers_file_path   = data_storage_folder_file_path + '/' + iter_specific_RR50_emp_stat_powers_file_name
        
            theo_placebo_arm_hists_file_iter_exists = os.path.isfile(iter_specific_theo_placebo_arm_hists_file_path)
            theo_drug_arm_hists_file_iter_exists    = os.path.isfile(iter_specific_theo_drug_arm_hists_file_path)
            RR50_emp_stat_powers_file_iter_exists   = os.path.isfile(iter_specific_RR50_emp_stat_powers_file_path)

            all_iter_specific_files_exist = theo_placebo_arm_hists_file_iter_exists and theo_drug_arm_hists_file_iter_exists and RR50_emp_stat_powers_file_iter_exists 

            each_iter_exists[compute_iter_index] = all_iter_specific_files_exist
    
        all_iter_files_exist = np.all(each_iter_exists)

        sleep(num_seconds_to_wait)
    
    command = ['sbatch', 'train_model_wrapper.sh', 
               str(monthly_mean_min), str(monthly_mean_max), str(monthly_std_dev_min), str(monthly_std_dev_max), 
               data_storage_folder_name, RR50_stat_power_model_file_name, str(num_compute_iters)]
    subprocess.call(command)