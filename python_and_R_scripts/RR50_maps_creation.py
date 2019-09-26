import numpy as np
from patient_population_generation import generate_homogenous_placebo_arm_patient_pop
from patient_population_generation import generate_homogenous_drug_arm_patient_pop
from endpoint_functions import calculate_percent_changes
import json
import os
import time
import psutil
import sys


def create_placebo_arm_percent_changes_from_homogenous_pop(num_patients_per_trial_arm,
                                                           monthly_mean, 
                                                           monthly_std_dev,
                                                           num_baseline_months,
                                                           num_testing_months,
                                                           minimum_required_baseline_seizure_count,
                                                           placebo_mu,
                                                           placebo_sigma):
    
    [monthly_placebo_arm_baseline_seizure_diaries, 
     monthly_placebo_arm_testing_seizure_diaries  ] = \
         generate_homogenous_placebo_arm_patient_pop(num_patients_per_trial_arm,
                                                     monthly_mean, 
                                                     monthly_std_dev,
                                                     num_baseline_months,
                                                     num_testing_months,
                                                     1,
                                                     1,
                                                     minimum_required_baseline_seizure_count,
                                                     placebo_mu,
                                                     placebo_sigma)
    
    placebo_arm_percent_changes = \
        calculate_percent_changes(monthly_placebo_arm_baseline_seizure_diaries,
                                  monthly_placebo_arm_testing_seizure_diaries,
                                  num_patients_per_trial_arm)
    
    return placebo_arm_percent_changes


def create_drug_arm_percent_changes_from_homogenous_pop(num_patients_per_trial_arm,
                                                        monthly_mean, 
                                                        monthly_std_dev,
                                                        num_baseline_months,
                                                        num_testing_months,
                                                        minimum_required_baseline_seizure_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma):
    
    [monthly_drug_arm_baseline_seizure_diaries, 
     monthly_drug_arm_testing_seizure_diaries  ] = \
         generate_homogenous_drug_arm_patient_pop(num_patients_per_trial_arm,
                                                  monthly_mean, 
                                                  monthly_std_dev,
                                                  num_baseline_months,
                                                  num_testing_months,
                                                  1,
                                                  1,
                                                  minimum_required_baseline_seizure_count,
                                                  placebo_mu,
                                                  placebo_sigma,
                                                  drug_mu,
                                                  drug_sigma)

    drug_arm_percent_changes = \
        calculate_percent_changes(monthly_drug_arm_baseline_seizure_diaries,
                                  monthly_drug_arm_testing_seizure_diaries,
                                  num_patients_per_trial_arm)

    return drug_arm_percent_changes


def estimate_expected_placebo_and_drug_RR50(num_patients_per_trial_arm,
                                            monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            num_testing_months,
                                            minimum_required_baseline_seizure_count,
                                            placebo_mu,
                                            placebo_sigma,
                                            drug_mu,
                                            drug_sigma,
                                            num_trials):

    placebo_RR50_array = np.zeros(num_trials)
    drug_RR50_array    = np.zeros(num_trials)

    for trial_index in range(num_trials):

        placebo_arm_percent_changes = \
            create_placebo_arm_percent_changes_from_homogenous_pop(num_patients_per_trial_arm,
                                                                   monthly_mean, 
                                                                   monthly_std_dev,
                                                                   num_baseline_months,
                                                                   num_testing_months,
                                                                   minimum_required_baseline_seizure_count,
                                                                   placebo_mu,
                                                                   placebo_sigma)

        drug_arm_percent_changes = \
            create_drug_arm_percent_changes_from_homogenous_pop(num_patients_per_trial_arm,
                                                                monthly_mean, 
                                                                monthly_std_dev,
                                                                num_baseline_months,
                                                                num_testing_months,
                                                                minimum_required_baseline_seizure_count,
                                                                placebo_mu,
                                                                placebo_sigma,
                                                                drug_mu,
                                                                drug_sigma)
                
        placebo_arm_RR50 = np.sum(placebo_arm_percent_changes > 0.5)/num_patients_per_trial_arm
        drug_arm_RR50    = np.sum(   drug_arm_percent_changes > 0.5)/num_patients_per_trial_arm

        placebo_RR50_array[trial_index] = placebo_arm_RR50
        drug_RR50_array[trial_index]    = drug_arm_RR50
            
    expected_placebo_RR50 = np.mean(placebo_RR50_array)
    expected_drug_RR50    = np.mean(drug_RR50_array)

    return [expected_placebo_RR50, expected_drug_RR50]


def estimate_expected_placebo_response_maps(monthly_mean_min,
                                            monthly_mean_max,
                                            monthly_std_dev_min,
                                            monthly_std_dev_max,
                                            num_patients_per_trial_arm,
                                            num_baseline_months,
                                            num_testing_months,
                                            minimum_required_baseline_seizure_count,
                                            placebo_mu,
                                            placebo_sigma,
                                            drug_mu,
                                            drug_sigma,
                                            num_trials):

    monthly_mean_array    = np.arange(monthly_mean_min, monthly_mean_max + 1)
    monthly_std_dev_array = np.arange(monthly_std_dev_min, monthly_std_dev_max + 1)

    num_monthly_means     = len(monthly_mean_array)
    num_monthly_std_devs  = len(monthly_std_dev_array)

    expected_placebo_response_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_drug_response_map    = np.zeros((num_monthly_std_devs, num_monthly_means))

    for monthly_mean_index in range(num_monthly_means):
        for monthly_std_dev_index in range(num_monthly_std_devs):
            
            monthly_mean    = monthly_mean_array[monthly_mean_index]
            monthly_std_dev = monthly_std_dev_array[monthly_std_dev_index]

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                [expected_placebo_RR50, expected_drug_RR50] = \
                    estimate_expected_placebo_and_drug_RR50(num_patients_per_trial_arm,
                                                            monthly_mean, 
                                                            monthly_std_dev,
                                                            num_baseline_months,
                                                            num_testing_months,
                                                            minimum_required_baseline_seizure_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            drug_mu,
                                                            drug_sigma,
                                                            num_trials)
            
                expected_placebo_response_map[monthly_std_dev_max - monthly_std_dev, monthly_mean_index] = expected_placebo_RR50
                expected_drug_response_map[monthly_std_dev_max - monthly_std_dev, monthly_mean_index]    = expected_drug_RR50
            
            else:

                expected_placebo_response_map[monthly_std_dev_max - monthly_std_dev, monthly_mean_index] = np.nan
                expected_drug_response_map[monthly_std_dev_max - monthly_std_dev, monthly_mean_index]    = np.nan
    
    return [expected_placebo_response_map, expected_drug_response_map]


def store_exected_RR50_maps(expected_RR50_map_storage_folder,
                            expected_placebo_response_map,
                            expected_drug_response_map):

    if( not os.path.isdir(expected_RR50_map_storage_folder) ):

        os.makedirs(expected_RR50_map_storage_folder)

    with open(expected_RR50_map_storage_folder + '/expected_RR50_placebo_arm_map.json', 'w+') as json_file:

        json.dump(expected_placebo_response_map.tolist(), json_file)
    
    with open(expected_RR50_map_storage_folder + '/expected_RR50_drug_arm_map.json',    'w+') as json_file:

        json.dump(expected_drug_response_map.tolist(), json_file)


if(__name__=='__main__'):

    monthly_mean_min = int(sys.argv[1])
    monthly_mean_max = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    num_patients_per_trial_arm = int(sys.argv[5])
    num_baseline_months = int(sys.argv[6])
    num_testing_months = int(sys.argv[7])
    minimum_required_baseline_seizure_count = int(sys.argv[8])

    placebo_mu = float(sys.argv[9])
    placebo_sigma = float(sys.argv[10])
    drug_mu = float(sys.argv[11])
    drug_sigma = float(sys.argv[12])

    num_trials = int(sys.argv[13])

    expected_RR50_map_storage_folder = sys.argv[14]

    RR50_map_creation_start_time_in_seconds = time.time()

    [expected_placebo_response_map, 
     expected_drug_response_map    ] = \
         estimate_expected_placebo_response_maps(monthly_mean_min,
                                                 monthly_mean_max,
                                                 monthly_std_dev_min,
                                                 monthly_std_dev_max,
                                                 num_patients_per_trial_arm,
                                                 num_baseline_months,
                                                 num_testing_months,
                                                 minimum_required_baseline_seizure_count,
                                                 placebo_mu,
                                                 placebo_sigma,
                                                 drug_mu,
                                                 drug_sigma,
                                                 num_trials)
    
    store_exected_RR50_maps(expected_RR50_map_storage_folder,
                            expected_placebo_response_map,
                            expected_drug_response_map)
    
    RR50_map_creation_stop_time_in_seconds = time.time()
    RR50_map_creation_runtime_in_seconds = RR50_map_creation_stop_time_in_seconds - RR50_map_creation_start_time_in_seconds
    RR50_map_creation_runtime_in_minutes = RR50_map_creation_runtime_in_seconds/60
    RR50_map_creation_runtime_in_minutes_str = str(np.round(RR50_map_creation_runtime_in_minutes, 3))

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)
    used_mem_in_gigabytes_str = str(np.round(used_mem_in_gigabytes, 3))

    print('\ntotal algorithm runtime: ' + RR50_map_creation_runtime_in_minutes_str + ' minutes\nnmemory used: ' + used_mem_in_gigabytes_str + ' GB\n')
