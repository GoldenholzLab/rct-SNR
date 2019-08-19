import numpy as np
import os
import json
import time
import sys
import psutil


def generate_one_trial_arm_of_patient_diaries(monthly_mean,
                                              monthly_std_dev,
                                              num_patients_per_dot, 
                                              min_req_base_sz_count,
                                              num_baseline_days_per_patient, 
                                              num_total_days_per_patient):

    daily_patient_diaries = np.zeros((num_patients_per_dot, num_total_days_per_patient))

    for patient_index in range(num_patients_per_dot):

        daily_mean = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)
        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)

        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        acceptable_baseline = False
        current_daily_patient_diary = np.zeros(num_total_days_per_patient)

        while(not acceptable_baseline):

            for day_index in range(num_total_days_per_patient):

                daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
                daily_count = np.random.poisson(daily_rate)

                current_daily_patient_diary[day_index] = daily_count
                
            current_patient_baseline_sz_count = np.sum(current_daily_patient_diary[:num_baseline_days_per_patient])
                
            if(current_patient_baseline_sz_count >= min_req_base_sz_count):

                    acceptable_baseline = True
            
        daily_patient_diaries[patient_index, :] = current_daily_patient_diary
    
    return daily_patient_diaries


def calculate_individual_patient_TTP_times_per_diary_set(daily_seizure_diaries,
                                                         num_baseline_days_per_patient,
                                                         num_testing_days_per_patient,
                                                         num_daily_seizure_diaries):

    baseline_daily_seizure_diaries = daily_seizure_diaries[:, :num_baseline_days_per_patient]
    testing_daily_seizure_diaries  = daily_seizure_diaries[:, num_baseline_days_per_patient:]

    baseline_monthly_seizure_frequencies = 28*np.mean(baseline_daily_seizure_diaries, 1)

    TTP_times      = np.zeros(num_daily_seizure_diaries)
    observed_array = np.zeros(num_daily_seizure_diaries)

    for daily_seizure_diary_index in range(num_daily_seizure_diaries):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries[daily_seizure_diary_index, day_index]

            reached_count  = sum_count >= baseline_monthly_seizure_frequencies[daily_seizure_diary_index]
            right_censored = day_index == (num_testing_days_per_patient - 1)
            reached_count  = reached_count or right_censored

            day_index = day_index + 1
        
        TTP_times[daily_seizure_diary_index]      = day_index
        observed_array[daily_seizure_diary_index] = not right_censored
    
    return [TTP_times, observed_array]


def apply_effect(daily_seizure_diaries,
                 num_baseline_days_per_patient,
                 num_testing_days_per_patient,
                 num_seizure_diaries,
                 effect_mu,
                 effect_sigma):
    
    testing_daily_seizure_diaries  = daily_seizure_diaries[:, num_baseline_days_per_patient:]

    for seizure_diary_index in range(num_seizure_diaries):

        effect = np.random.normal(effect_mu, effect_sigma)
        if(effect > 1):
            effect = 1
        
        current_testing_daily_seizure_diary = testing_daily_seizure_diaries[seizure_diary_index, :]

        for day_index in range(num_testing_days_per_patient):

            current_seizure_count = current_testing_daily_seizure_diary[day_index]
            num_removed = 0

            for seizure_index in range(np.int_(current_seizure_count)):

                if(np.random.random() <= np.abs(effect)):

                    num_removed = num_removed + np.sign(effect)
                
            current_seizure_count = current_seizure_count - num_removed
            current_testing_daily_seizure_diary[day_index] = current_seizure_count
        
        testing_daily_seizure_diaries[seizure_diary_index, :] = current_testing_daily_seizure_diary

    daily_seizure_diaries[:, num_baseline_days_per_patient:] = testing_daily_seizure_diaries

    return daily_seizure_diaries


def generate_individual_patient_TTP_times_per_trial_arm(monthly_mean,
                                                        monthly_std_dev,
                                                        num_patients_per_dot,
                                                        num_baseline_days_per_patient,
                                                        num_testing_days_per_patient,
                                                        num_total_days_per_patient,
                                                        min_req_base_sz_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma):

    one_trial_arm_daily_seizure_diaries = \
        generate_one_trial_arm_of_patient_diaries(monthly_mean,
                                                  monthly_std_dev, 
                                                  num_patients_per_dot, 
                                                  min_req_base_sz_count,
                                                  num_baseline_days_per_patient, 
                                                  num_total_days_per_patient)

    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_patients_per_dot,
                     placebo_mu,
                     placebo_sigma)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_patients_per_dot,
                     drug_mu,
                     drug_sigma)

    [one_trial_arm_TTP_times, one_trial_arm_observed_array] = \
        calculate_individual_patient_TTP_times_per_diary_set(one_trial_arm_daily_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_patients_per_dot)

    return [one_trial_arm_TTP_times, one_trial_arm_observed_array]


def generate_one_trial_TTP_times(monthly_mean,
                                 monthly_std_dev,
                                 min_req_base_sz_count,
                                 num_baseline_days_per_patient,
                                 num_testing_days_per_patient,
                                 num_total_days_per_patient,
                                 num_patients_per_dot,
                                 placebo_mu,
                                 placebo_sigma,
                                 drug_mu,
                                 drug_sigma):

    [one_placebo_arm_TTP_times, one_placebo_arm_observed_array] = \
        generate_individual_patient_TTP_times_per_trial_arm(monthly_mean,
                                                            monthly_std_dev,
                                                            num_patients_per_dot,
                                                            num_baseline_days_per_patient,
                                                            num_testing_days_per_patient,
                                                            num_total_days_per_patient,
                                                            min_req_base_sz_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            0,
                                                            0)
    
    [one_drug_arm_TTP_times, one_drug_arm_observed_array] = \
        generate_individual_patient_TTP_times_per_trial_arm(monthly_mean,
                                                            monthly_std_dev, 
                                                            num_patients_per_dot,
                                                            num_baseline_days_per_patient,
                                                            num_testing_days_per_patient,
                                                            num_total_days_per_patient,
                                                            min_req_base_sz_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            drug_mu,
                                                            drug_sigma)
    
    return [one_placebo_arm_TTP_times, one_placebo_arm_observed_array, 
            one_drug_arm_TTP_times,    one_drug_arm_observed_array   ]


def generate_one_trial_median_TTP_times_per_arm(monthly_mean,
                                                monthly_std_dev,
                                                min_req_base_sz_count,
                                                num_baseline_days_per_patient,
                                                num_testing_days_per_patient,
                                                num_total_days_per_patient,
                                                num_patients_per_dot,
                                                placebo_mu,
                                                placebo_sigma,
                                                drug_mu,
                                                drug_sigma):

    start_time_in_seconds = time.time()

    [one_placebo_arm_TTP_times, _, 
     one_drug_arm_TTP_times,    _] = \
         generate_one_trial_TTP_times(monthly_mean,
                                      monthly_std_dev,
                                      min_req_base_sz_count,
                                      num_baseline_days_per_patient,
                                      num_testing_days_per_patient,
                                      num_total_days_per_patient,
                                      num_patients_per_dot,
                                      placebo_mu,
                                      placebo_sigma,
                                      drug_mu,
                                      drug_sigma)

    num_placebo_arm_median_TTP_time = np.median(one_placebo_arm_TTP_times)
    num_drug_arm_median_TTP_time    = np.median(one_drug_arm_TTP_times)

    num_placebo_arm_median_TTP_time_occurrences = np.sum(one_placebo_arm_TTP_times == num_placebo_arm_median_TTP_time)
    num_drug_arm_median_TTP_time_occurrences    = np.sum(one_placebo_arm_TTP_times == num_drug_arm_median_TTP_time)

    stop_time_in_seconds = time.time()
    total_runtime_in_seconds = stop_time_in_seconds - start_time_in_seconds
    total_runtime_in_minutes = total_runtime_in_seconds/60
    total_runtime_in_minutes_str = str(np.round(total_runtime_in_minutes, 3))
    
    print('\n[monthly mean, monthly standard deviation]: ' + str([monthly_mean, monthly_std_dev]) + '\ntotal runtime: ' + total_runtime_in_minutes_str + ' minutes\n')

    return [num_placebo_arm_median_TTP_time, num_placebo_arm_median_TTP_time_occurrences,
            num_drug_arm_median_TTP_time,    num_drug_arm_median_TTP_time_occurrences   ]


def create_median_TTP_time_map_per_trial_arm(monthly_mean_min,
                                             monthly_mean_max,
                                             monthly_std_dev_min,
                                             monthly_std_dev_max,
                                             min_req_base_sz_count,
                                             num_baseline_days_per_patient,
                                             num_testing_days_per_patient,
                                             num_total_days_per_patient,
                                             num_patients_per_dot,
                                             placebo_mu,
                                             placebo_sigma,
                                             drug_mu,
                                             drug_sigma):
    
    monthly_mean_array    = np.arange(monthly_mean_min,    monthly_mean_max    + 1)
    monthly_std_dev_array = np.arange(monthly_std_dev_min, monthly_std_dev_max + 1)
    num_monthly_means     = len(monthly_mean_array)
    num_monthly_std_dev   = len(monthly_std_dev_array)

    placebo_arm_median_TTP_time_map = np.zeros((num_monthly_std_dev, num_monthly_means))
    drug_arm_median_TTP_time_map    = np.zeros((num_monthly_std_dev, num_monthly_means))

    for monthly_mean_index in range(num_monthly_means):
        for monthly_std_dev_index in range(num_monthly_std_dev):

            monthly_mean    = monthly_mean_array[monthly_mean_index]
            monthly_std_dev = monthly_std_dev_max - monthly_std_dev_array[monthly_std_dev_index] + 1

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                [num_placebo_arm_median_TTP_time, _,
                 num_drug_arm_median_TTP_time,    _] = \
                     generate_one_trial_median_TTP_times_per_arm(monthly_mean,
                                                                 monthly_std_dev,
                                                                 min_req_base_sz_count,
                                                                 num_baseline_days_per_patient,
                                                                 num_testing_days_per_patient,
                                                                 num_total_days_per_patient,
                                                                 num_patients_per_dot,
                                                                 placebo_mu,
                                                                 placebo_sigma,
                                                                 drug_mu,
                                                                 drug_sigma)

                placebo_arm_median_TTP_time_map[monthly_std_dev_index, monthly_mean_index] = num_placebo_arm_median_TTP_time
                drug_arm_median_TTP_time_map[monthly_std_dev_index, monthly_mean_index]    = num_drug_arm_median_TTP_time
            
            else:

                placebo_arm_median_TTP_time_map[monthly_std_dev_index, monthly_mean_index] = np.nan
                drug_arm_median_TTP_time_map[monthly_std_dev_index, monthly_mean_index]    = np.nan
    
    return [placebo_arm_median_TTP_time_map, drug_arm_median_TTP_time_map]


def store_map(map, map_file_name, map_folder):

    map_file_path = map_folder + '/' + map_file_name + '.map'
    with open(map_file_path, 'w+') as json_file: 
        json.dump(map.tolist(), json_file)


if(__name__=='__main__'):

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    min_req_base_sz_count           = int(sys.argv[5])
    num_baseline_months_per_patient = int(sys.argv[6])
    num_testing_months_per_patient  = int(sys.argv[7])
    num_patients_per_dot            = int(sys.argv[8])

    placebo_mu    = float(sys.argv[9])
    placebo_sigma = float(sys.argv[10])
    drug_mu       = float(sys.argv[11])
    drug_sigma    = float(sys.argv[12])

    placebo_arm_median_TTP_time_map_file_name = sys.argv[13]
    drug_arm_median_TTP_time_map_file_name    = sys.argv[14]

    algorithm_start_time_in_seconds = time.time()

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    [placebo_arm_median_TTP_time_map, drug_arm_median_TTP_time_map] = \
        create_median_TTP_time_map_per_trial_arm(monthly_mean_min,
                                                 monthly_mean_max,
                                                 monthly_std_dev_min,
                                                 monthly_std_dev_max,
                                                 min_req_base_sz_count,
                                                 num_baseline_days_per_patient,
                                                 num_testing_days_per_patient,
                                                 num_total_days_per_patient,
                                                 num_patients_per_dot,
                                                 placebo_mu,
                                                 placebo_sigma,
                                                 drug_mu,
                                                 drug_sigma)

    store_map(placebo_arm_median_TTP_time_map, placebo_arm_median_TTP_time_map_file_name, os.getcwd())
    store_map(drug_arm_median_TTP_time_map,    drug_arm_median_TTP_time_map_file_name,    os.getcwd())

    algorithm_stop_time_in_seconds = time.time()

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)
    used_mem_in_gigabytes_str = str(np.round(used_mem_in_gigabytes, 3))

    total_algorithm_runtime_in_seconds = algorithm_stop_time_in_seconds - algorithm_start_time_in_seconds
    total_algorithm_runtime_in_minutes = total_algorithm_runtime_in_seconds/60
    total_algorithm_runtime_in_minutes_str = str(np.round(total_algorithm_runtime_in_minutes, 3))
    print('\ntotal algorithm runtime: ' + total_algorithm_runtime_in_minutes_str + ' minutes\nmemory used: ' + used_mem_in_gigabytes_str + ' GB\nWZ')



