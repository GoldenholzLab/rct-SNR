from patient_population_generation import generate_theo_patient_pop_params
from RR50_map_stat_power_prediction import estimate_power_of_theo_pop
from empirical_estimation import empirically_estimate_RR50_statistical_power
import os
import json
import numpy as np
import time
import psutil

if(__name__=='__main__'):

    monthly_mean_min = 1
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16
    num_theo_patients_per_trial_arm = 153

    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4
    num_trials = 10

    placebo_mu = 0
    placebo_sigma = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05

    maps_folder = '/Users/juanromero/Documents/Github/rct-SNR/RR50_test_folder'
    RMSE_folder = '/Users/juanromero/Documents/Github/rct-SNR/RR50_RMSE_folder'
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

    RR50_emp_stat_power = \
        empirically_estimate_RR50_statistical_power(theo_placebo_arm_patient_pop_params,
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

    fisher_exact_stat_power = \
        estimate_power_of_theo_pop(maps_folder,
                                   monthly_mean_min,
                                   monthly_mean_max,
                                   monthly_std_dev_min,
                                   monthly_std_dev_max,
                                   theo_placebo_arm_patient_pop_params,
                                   theo_drug_arm_patient_pop_params,
                                   num_theo_patients_per_trial_arm)
    
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

    print('\n' + str([100*fisher_exact_stat_power, 100*RR50_emp_stat_power]) + '\n')

    if( not os.path.isdir(RMSE_folder) ):
        os.makedirs(RMSE_folder)

    with open(RMSE_folder + '/' + str(estim_iter) + '.json', 'w+') as json_file:

        json.dump([fisher_exact_stat_power, RR50_emp_stat_power], json_file)
