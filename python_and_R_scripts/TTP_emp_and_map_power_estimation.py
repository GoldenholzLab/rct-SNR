import os
import json
import numpy as np
from patient_population_generation import generate_theo_patient_pop_params
from empirical_estimation import empirically_estimate_TTP_statistical_power
from TTP_map_stat_power_prediction import estimate_map_based_stat_power_for_one_pop
import time
import psutil


if(__name__=='__main__'):

    monthly_mean_min = 4
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8
    num_theo_patients_per_trial_arm = 153
    
    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4
    placebo_mu = 0
    placebo_sigma = 0.05 
    drug_mu = 0.2
    drug_sigma = 0.05
    num_trials = 500

    num_patients_per_map_location = 5000
    alpha = 0.05

    folder = os.getcwd() + '/test_folder'
    RMSE_folder = os.getcwd()
    estim_iter = 1

    theo_placebo_arm_patient_pop_params = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    theo_drug_arm_patient_pop_params = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    empirical_start_time_in_seconds = time.time()

    TTP_emp_stat_power = \
        empirically_estimate_TTP_statistical_power(theo_placebo_arm_patient_pop_params,
                                                   theo_drug_arm_patient_pop_params,
                                                   num_theo_patients_per_trial_arm,
                                                   num_baseline_months,
                                                   num_testing_months,
                                                   minimum_required_baseline_seizure_count,
                                                   placebo_mu,
                                                   placebo_sigma,
                                                   drug_mu,
                                                   drug_sigma,
                                                   num_trials)

    empirical_stop_time_in_seconds = time.time()
    empirical_total_runtime_in_seconds = empirical_stop_time_in_seconds - empirical_start_time_in_seconds
    empirical_total_runtime_in_minutes = empirical_total_runtime_in_seconds/60
    empirical_total_runtime_in_minutes_str = str(np.round(empirical_total_runtime_in_minutes, 3))

    print('\nempirical runtime: ' + empirical_total_runtime_in_minutes_str + ' minutes\n')

    map_start_time_in_seconds = time.time()

    map_stat_power = \
        estimate_map_based_stat_power_for_one_pop(folder,
                                                  num_theo_patients_per_trial_arm,
                                                  num_patients_per_map_location,
                                                  str(estim_iter),
                                                  alpha,
                                                  theo_placebo_arm_patient_pop_params,
                                                  theo_drug_arm_patient_pop_params)

    map_stop_time_in_seconds = time.time()
    map_total_runtime_in_seconds = map_stop_time_in_seconds - map_start_time_in_seconds
    map_total_runtime_in_minutes = map_total_runtime_in_seconds/60
    map_total_runtime_in_minutes_str = str(np.round(map_total_runtime_in_minutes, 3))

    print('\nmap-based runtime: ' + map_total_runtime_in_minutes_str + ' minutes\n')

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)
    used_mem_in_gigabytes_str = str(np.round(used_mem_in_gigabytes, 3))

    print('\nmemory usage: ' + used_mem_in_gigabytes_str + ' GB\n')
    
    print('\n' + str([100*map_stat_power, 100*TTP_emp_stat_power]) + '\n')

    if( not os.path.isdir(RMSE_folder) ):
        os.makedirs(RMSE_folder)

    with open(RMSE_folder + '/' + str(estim_iter) + '.json', 'w+') as json_file:

        json.dump([map_stat_power, TTP_emp_stat_power], json_file)
