import os
import numpy as np
from time import sleep
import subprocess
import sys


if(__name__=='__main__'):

    data_storage_folder_name    =  sys.argv[1]
    num_compute_iters        = int(sys.argv[2])

    data_storage_folder_file_path = './' + data_storage_folder_name

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

    print(all_iter_files_exist)
    