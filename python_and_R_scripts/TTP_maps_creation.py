import numpy as np
from patient_population_generation import generate_homogenous_drug_arm_patient_pop
from patient_population_generation import generate_homogenous_placebo_arm_patient_pop
from endpoint_functions import calculate_time_to_prerandomizations
import json
import time
import psutil


def create_placebo_arm_TTP_times_from_homogenous_pop(num_patients_per_map_location,
                                                     monthly_mean, 
                                                     monthly_std_dev,
                                                     num_baseline_months,
                                                     num_testing_months,
                                                     num_testing_days,
                                                     baseline_time_scaling_const,
                                                     testing_time_scaling_const,
                                                     minimum_required_baseline_seizure_count,
                                                     placebo_mu,
                                                     placebo_sigma):

    [monthly_placebo_arm_baseline_seizure_diaries, 
     daily_placebo_arm_testing_seizure_diaries  ] = \
         generate_homogenous_placebo_arm_patient_pop(num_patients_per_map_location,
                                                     monthly_mean, 
                                                     monthly_std_dev,
                                                     num_baseline_months,
                                                     num_testing_months,
                                                     baseline_time_scaling_const,
                                                     testing_time_scaling_const,
                                                     minimum_required_baseline_seizure_count,
                                                     placebo_mu,
                                                     placebo_sigma)

    [placebo_arm_TTP_times, placebo_arm_observed_array] = \
        calculate_time_to_prerandomizations(monthly_placebo_arm_baseline_seizure_diaries,
                                            daily_placebo_arm_testing_seizure_diaries,
                                            num_patients_per_map_location,
                                            num_testing_days)
    
    return [placebo_arm_TTP_times, placebo_arm_observed_array]


def create_drug_arm_TTP_times_from_homogenous_pop(num_patients_per_map_location,
                                                  monthly_mean, 
                                                  monthly_std_dev,
                                                  num_baseline_months,
                                                  num_testing_months,
                                                  num_testing_days,
                                                  baseline_time_scaling_const,
                                                  testing_time_scaling_const,
                                                  minimum_required_baseline_seizure_count,
                                                  placebo_mu,
                                                  placebo_sigma,
                                                  drug_mu,
                                                  drug_sigma):

    [monthly_drug_arm_baseline_seizure_diaries, 
     daily_drug_arm_testing_seizure_diaries  ] = \
         generate_homogenous_drug_arm_patient_pop(num_patients_per_map_location,
                                                  monthly_mean, 
                                                  monthly_std_dev,
                                                  num_baseline_months,
                                                  num_testing_months,
                                                  baseline_time_scaling_const,
                                                  testing_time_scaling_const,
                                                  minimum_required_baseline_seizure_count,
                                                  placebo_mu,
                                                  placebo_sigma,
                                                  drug_mu,
                                                  drug_sigma)

    [drug_arm_TTP_times, drug_arm_observed_array] = \
        calculate_time_to_prerandomizations(monthly_drug_arm_baseline_seizure_diaries,
                                            daily_drug_arm_testing_seizure_diaries,
                                            num_patients_per_map_location,
                                            num_testing_days)
    
    return [drug_arm_TTP_times, drug_arm_observed_array]


def create_TTP_hist_maps(monthly_mean_min, 
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_patients_per_map_location,
                         num_baseline_months,
                         num_testing_months,
                         baseline_time_scaling_const,
                         testing_time_scaling_const,
                         minimum_required_baseline_seizure_count,
                         placebo_mu,
                         placebo_sigma,
                         drug_mu,
                         drug_sigma):

    monthly_mean_array    = np.arange(monthly_mean_min, monthly_mean_max + 1)
    monthly_std_dev_array = np.arange(monthly_std_dev_min, monthly_std_dev_max + 1)

    num_monthly_means     = len(monthly_mean_array)
    num_monthly_std_devs  = len(monthly_std_dev_array)
    num_testing_days = num_testing_months*testing_time_scaling_const

    placebo_hist_array    = np.zeros((num_monthly_std_devs, num_monthly_means, num_testing_days))
    drug_hist_array       = np.zeros((num_monthly_std_devs, num_monthly_means, num_testing_days))

    for monthly_mean_index in range(num_monthly_means):
        for monthly_std_dev_index in range(num_monthly_std_devs):
            
            monthly_mean    = monthly_mean_array[monthly_mean_index]
            monthly_std_dev = monthly_std_dev_array[monthly_std_dev_index]
            
            point_start_time_in_seconds = time.time()

            try:

                [placebo_arm_TTP_times, _] = \
                    create_placebo_arm_TTP_times_from_homogenous_pop(num_patients_per_map_location,
                                                                     monthly_mean, 
                                                                     monthly_std_dev,
                                                                     num_baseline_months,
                                                                     num_testing_months,
                                                                     num_testing_days,
                                                                     baseline_time_scaling_const,
                                                                     testing_time_scaling_const,
                                                                     minimum_required_baseline_seizure_count,
                                                                     placebo_mu,
                                                                     placebo_sigma)
                
                [drug_arm_TTP_times, _] = \
                    create_drug_arm_TTP_times_from_homogenous_pop(num_patients_per_map_location,
                                                                  monthly_mean, 
                                                                  monthly_std_dev,
                                                                  num_baseline_months,
                                                                  num_testing_months,
                                                                  num_testing_days,
                                                                  baseline_time_scaling_const,
                                                                  testing_time_scaling_const,
                                                                  minimum_required_baseline_seizure_count,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  drug_mu,
                                                                  drug_sigma)

                [placebo_TTP_hist, _] = np.histogram(placebo_arm_TTP_times, bins=num_testing_days, range=[1, num_testing_days + 1])

                [drug_TTP_hist, _] = np.histogram(drug_arm_TTP_times, bins=num_testing_days, range=[1, num_testing_days + 1])

                placebo_hist_array[monthly_mean_index, monthly_std_dev_index, :] = placebo_TTP_hist
                drug_hist_array[monthly_mean_index, monthly_std_dev_index, :] = drug_TTP_hist

            except ValueError as error:

                zero_value_mean = error.args[0] == 'The true monthly mean of a homogenous patient population cannot be zero.'
                no_overdispersion = error.args[0] == 'The monthly standard deviation must be greater than the square root of the monthly mean for a homogenous patient population.'
                
                if((not zero_value_mean) and (not no_overdispersion)):

                    raise

                else:

                    placebo_hist_array[monthly_mean_index, monthly_std_dev_index, :] = np.array(num_testing_days*[np.nan])
                    drug_hist_array[monthly_mean_index, monthly_std_dev_index, :]    = np.array(num_testing_days*[np.nan])
            
            point_stop_time_in_seconds = time.time()
            point_total_runtime_in_seconds = point_stop_time_in_seconds - point_start_time_in_seconds
            point_total_runtime_in_minutes = point_total_runtime_in_seconds/60
            point_total_runtime_in_minutes_str = str(np.round(point_total_runtime_in_minutes, 3))
            
            print('\n[monthly mean, monthly standard deviation]: ' + str([monthly_mean, monthly_std_dev]) + '\npoint runtime: ' + point_total_runtime_in_minutes_str + ' minutes\n')

    return [placebo_hist_array, drug_hist_array]


def store_TTP_hists(folder, placebo_hist_array, drug_hist_array):

    with open(folder + '/placebo_hist_array.json', 'w+') as json_file:

        json.dump(placebo_hist_array.tolist(), json_file)

    with open(folder + '/drug_hist_array.json', 'w+') as json_file:

        json.dump(drug_hist_array.tolist(), json_file)


if(__name__=='__main__'):

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_patients_per_map_location = 5000
    num_baseline_months = 2
    num_testing_months = 3
    baseline_time_scaling_const = 1
    testing_time_scaling_const = 28
    minimum_required_baseline_seizure_count = 4
    placebo_mu = 0
    placebo_sigma = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05

    import os
    folder = os.getcwd()

    algorithm_start_time_in_seconds = time.time()

    [placebo_hist_array, drug_hist_array] = \
        create_TTP_hist_maps(monthly_mean_min, 
                             monthly_mean_max,
                             monthly_std_dev_min,
                             monthly_std_dev_max,
                             num_patients_per_map_location,
                             num_baseline_months,
                             num_testing_months,
                             baseline_time_scaling_const,
                             testing_time_scaling_const,
                             minimum_required_baseline_seizure_count,
                             placebo_mu,
                             placebo_sigma,
                             drug_mu,
                             drug_sigma)
    
    algorithm_stop_time_in_seconds = time.time()
    algorithm_total_runtime_in_seconds = algorithm_stop_time_in_seconds - algorithm_start_time_in_seconds
    algorithm_total_runtime_in_minutes = algorithm_total_runtime_in_seconds/60
    algorithm_total_runtime_in_minutes_str = str(np.round(algorithm_total_runtime_in_minutes, 3))

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)
    used_mem_in_gigabytes_str = str(np.round(used_mem_in_gigabytes, 3))
    
    print('\ntotal algorithm runtime: ' + algorithm_total_runtime_in_minutes_str + ' minutes\nnmemory used: ' + used_mem_in_gigabytes_str + ' GB\n')

    store_TTP_hists(folder, placebo_hist_array, drug_hist_array)

