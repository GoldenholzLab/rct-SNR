import sys
import numpy as np
import subprocess
import pandas as pd
import os
import time
import json


def generate_one_trial_arm_of_patient_diaries(monthly_mean,
                                              monthly_std_dev,
                                              num_theo_patients_per_trial_arm, 
                                              min_req_base_sz_count,
                                              num_baseline_days_per_patient, 
                                              num_total_days_per_patient):

    daily_patient_diaries = np.zeros((num_theo_patients_per_trial_arm, num_total_days_per_patient))

    for patient_index in range(num_theo_patients_per_trial_arm):

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
                                                        num_theo_patients_per_trial_arm,
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
                                                  num_theo_patients_per_trial_arm, 
                                                  min_req_base_sz_count,
                                                  num_baseline_days_per_patient, 
                                                  num_total_days_per_patient)

    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_theo_patients_per_trial_arm,
                     placebo_mu,
                     placebo_sigma)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_theo_patients_per_trial_arm,
                     drug_mu,
                     drug_sigma)

    [one_trial_arm_TTP_times, one_trial_arm_observed_array] = \
        calculate_individual_patient_TTP_times_per_diary_set(one_trial_arm_daily_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_theo_patients_per_trial_arm)

    return [one_trial_arm_TTP_times, one_trial_arm_observed_array]


def generate_one_trial_TTP_times(monthly_mean,
                                 monthly_std_dev,
                                 min_req_base_sz_count,
                                 num_baseline_days_per_patient,
                                 num_testing_days_per_patient,
                                 num_total_days_per_patient,
                                 num_theo_patients_per_trial_arm,
                                 placebo_mu,
                                 placebo_sigma,
                                 drug_mu,
                                 drug_sigma):

    [one_placebo_arm_TTP_times, one_placebo_arm_observed_array] = \
        generate_individual_patient_TTP_times_per_trial_arm(monthly_mean,
                                                            monthly_std_dev,
                                                            num_theo_patients_per_trial_arm,
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
                                                            num_theo_patients_per_trial_arm,
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


def generate_one_trial_analytical_quantities(one_placebo_arm_TTP_times,
                                             one_placebo_arm_observed_array,
                                             one_drug_arm_TTP_times,
                                             one_drug_arm_observed_array,
                                             num_theo_patients_per_trial_arm,
                                             tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(one_placebo_arm_TTP_times, one_drug_arm_TTP_times)
    events                 = np.append(one_placebo_arm_observed_array, one_drug_arm_observed_array)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*['C']) , np.array(num_theo_patients_per_trial_arm*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'estimate_log_hazard_ratio.R', relative_tmp_file_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])
    os.remove(relative_tmp_file_path)

    prob_fail_placebo_arm = np.sum(one_placebo_arm_observed_array == True)/num_theo_patients_per_trial_arm
    prob_fail_drug_arm    = np.sum(one_drug_arm_observed_array == True)/num_theo_patients_per_trial_arm

    return [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm]


def generate_one_map_point_analytical_quantities(monthly_mean, 
                                                 monthly_std_dev,
                                                 min_req_base_sz_count,
                                                 num_baseline_days_per_patient,
                                                 num_testing_days_per_patient,
                                                 num_total_days_per_patient,
                                                 num_theo_patients_per_trial_arm,
                                                 placebo_mu,
                                                 placebo_sigma,
                                                 drug_mu,
                                                 drug_sigma,
                                                 num_trials):

    tmp_file_name = str(monthly_mean) + '_' + str(monthly_std_dev)

    postulated_log_hazard_ratio_array = np.zeros(num_trials)
    prob_fail_placebo_arm_array       = np.zeros(num_trials)
    prob_fail_drug_arm_array          = np.zeros(num_trials)

    for trial_index in range(num_trials):

        trial_start_time_in_seconds = time.time()

        [one_placebo_arm_TTP_times, one_placebo_arm_observed_array, 
         one_drug_arm_TTP_times,    one_drug_arm_observed_array   ] = \
            generate_one_trial_TTP_times(monthly_mean, 
                                         monthly_std_dev,
                                         min_req_base_sz_count,
                                         num_baseline_days_per_patient,
                                         num_testing_days_per_patient,
                                         num_total_days_per_patient,
                                         num_theo_patients_per_trial_arm,
                                         placebo_mu,
                                         placebo_sigma,
                                         drug_mu,
                                         drug_sigma)

        [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm] = \
            generate_one_trial_analytical_quantities(one_placebo_arm_TTP_times,
                                                     one_placebo_arm_observed_array,
                                                     one_drug_arm_TTP_times,
                                                     one_drug_arm_observed_array,
                                                     num_theo_patients_per_trial_arm,
                                                     tmp_file_name)
            
        postulated_log_hazard_ratio_array[trial_index] = postulated_log_hazard_ratio
        prob_fail_placebo_arm_array[trial_index]       = prob_fail_placebo_arm
        prob_fail_drug_arm_array[trial_index]          = prob_fail_drug_arm

        trial_total_runtime_in_seconds_str = str(np.round(time.time() - trial_start_time_in_seconds, 2))

        print('trial #: ' + str(trial_index + 1) + ', runtime: ' + trial_total_runtime_in_seconds_str + ' seconds')
        
    average_postulated_log_hazard_ratio = np.mean(postulated_log_hazard_ratio_array)
    average_prob_fail_placebo_arm       = np.mean(prob_fail_placebo_arm_array)
    average_prob_fail_drug_arm          = np.mean(prob_fail_drug_arm_array)
        
    return [average_postulated_log_hazard_ratio, average_prob_fail_placebo_arm, average_prob_fail_drug_arm]


if(__name__=='__main__'):

    monthly_mean                    =   int(sys.argv[1])
    monthly_std_dev                 =   int(sys.argv[2])
    min_req_base_sz_count           =   int(sys.argv[3])
    num_baseline_months_per_patient =   int(sys.argv[4])
    num_testing_months_per_patient  =   int(sys.argv[5])
    num_theo_patients_per_trial_arm =   int(sys.argv[6])
    placebo_mu                      = float(sys.argv[7])
    placebo_sigma                   = float(sys.argv[8])
    drug_mu                         = float(sys.argv[9])
    drug_sigma                      = float(sys.argv[10])
    num_trials                      =   int(sys.argv[11])

    [average_postulated_log_hazard_ratio, average_prob_fail_placebo_arm, average_prob_fail_drug_arm] = \
        generate_one_map_point_analytical_quantities(monthly_mean, 
                                                     monthly_std_dev,
                                                     min_req_base_sz_count,
                                                     num_baseline_months_per_patient,
                                                     num_testing_months_per_patient,
                                                     num_theo_patients_per_trial_arm,
                                                     placebo_mu,
                                                     placebo_sigma,
                                                     drug_mu,
                                                     drug_sigma,
                                                     num_trials)


'''
def generate_one_map_point_analytical_quantities(monthly_mean, 
                                                 monthly_std_dev,
                                                 min_req_base_sz_count,
                                                 num_baseline_days_per_patient,
                                                 num_testing_days_per_patient,
                                                 num_total_days_per_patient,
                                                 num_theo_patients_per_trial_arm,
                                                 placebo_mu,
                                                 placebo_sigma,
                                                 drug_mu,
                                                 drug_sigma,
                                                 num_trials):

    print('\n[monthly mean, monthly standard deviation]: [' + str(monthly_mean) + ', ' + str(monthly_std_dev) + ']')

    if(monthly_std_dev > np.sqrt(monthly_mean)):

        tmp_file_name = str(monthly_mean) + '_' + str(monthly_std_dev)

        postulated_log_hazard_ratio_array = np.zeros(num_trials)
        prob_fail_placebo_arm_array       = np.zeros(num_trials)
        prob_fail_drug_arm_array          = np.zeros(num_trials)

        for trial_index in range(num_trials):

            trial_start_time_in_seconds = time.time()

            [one_placebo_arm_TTP_times, one_placebo_arm_observed_array, 
             one_drug_arm_TTP_times,    one_drug_arm_observed_array   ] = \
                generate_one_trial_TTP_times(monthly_mean, 
                                             monthly_std_dev,
                                             min_req_base_sz_count,
                                             num_baseline_days_per_patient,
                                             num_testing_days_per_patient,
                                             num_total_days_per_patient,
                                             num_theo_patients_per_trial_arm,
                                             placebo_mu,
                                             placebo_sigma,
                                             drug_mu,
                                             drug_sigma)

            [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm] = \
                generate_one_trial_analytical_quantities(one_placebo_arm_TTP_times,
                                                         one_placebo_arm_observed_array,
                                                         one_drug_arm_TTP_times,
                                                         one_drug_arm_observed_array,
                                                         num_theo_patients_per_trial_arm,
                                                         tmp_file_name)
            
            postulated_log_hazard_ratio_array[trial_index] = postulated_log_hazard_ratio
            prob_fail_placebo_arm_array[trial_index]       = prob_fail_placebo_arm
            prob_fail_drug_arm_array[trial_index]          = prob_fail_drug_arm

            trial_total_runtime_in_seconds_str = str(np.round(time.time() - trial_start_time_in_seconds, 2))

            print('trial #: ' + str(trial_index + 1) + ', runtime: ' + trial_total_runtime_in_seconds_str + ' seconds')
        
        average_postulated_log_hazard_ratio = np.mean(postulated_log_hazard_ratio_array)
        average_prob_fail_placebo_arm       = np.mean(prob_fail_placebo_arm_array)
        average_prob_fail_drug_arm          = np.mean(prob_fail_drug_arm_array)
        
        return [average_postulated_log_hazard_ratio, average_prob_fail_placebo_arm, average_prob_fail_drug_arm]
    
    else:

        return [np.nan, np.nan, np.nan]


def generate_maps(monthly_mean_min,
                  monthly_mean_max,
                  monthly_std_dev_min,
                  monthly_std_dev_max,
                  num_baseline_months_per_patient,
                  num_testing_months_per_patient,
                  min_req_base_sz_count,
                  num_theo_patients_per_trial_arm,
                  placebo_mu,
                  placebo_sigma,
                  drug_mu,
                  drug_sigma,
                  num_trials):

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    num_monthly_means     = monthly_mean_max - monthly_mean_min + 1
    num_monthly_std_devs = monthly_std_dev_max - monthly_std_dev_min + 1
    monthly_mean_array    = np.arange(monthly_mean_min,    monthly_mean_max + 1)
    monthly_std_dev_array = np.arange(monthly_std_dev_min, monthly_std_dev_max + 1)
    
    average_log_hazard_ratio_map      = np.zeros((num_monthly_std_devs, num_monthly_means))
    average_prob_fail_placebo_arm_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    average_prob_fail_drug_arm_map    = np.zeros((num_monthly_std_devs, num_monthly_means))

    for monthly_mean in monthly_mean_array:
        for monthly_std_dev in monthly_std_dev_array:

            [average_postulated_log_hazard_ratio, average_prob_fail_placebo_arm, average_prob_fail_drug_arm] = \
                generate_one_map_point_analytical_quantities(monthly_mean, 
                                                             monthly_std_dev,
                                                             min_req_base_sz_count,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_total_days_per_patient,
                                                             num_theo_patients_per_trial_arm,
                                                             placebo_mu,
                                                             placebo_sigma,
                                                             drug_mu,
                                                             drug_sigma,
                                                             num_trials)
           
            average_log_hazard_ratio_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min]      = average_postulated_log_hazard_ratio
            average_prob_fail_placebo_arm_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min] = average_prob_fail_placebo_arm
            average_prob_fail_drug_arm_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min]    = average_prob_fail_drug_arm
    
    return [average_log_hazard_ratio_map, average_prob_fail_placebo_arm_map, average_prob_fail_drug_arm_map]


if(__name__=='__main__'):

    monthly_mean_min    = 6
    monthly_mean_max    = 8
    monthly_std_dev_min = 6
    monthly_std_dev_max = 8

    num_theo_patients_per_trial_arm = 150
    min_req_base_sz_count           = 4
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    num_trials                      = 2

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    average_log_hazard_ratio_map_file_name      = 'average_log_hazard_ratio_map_file_name'
    average_prob_fail_placebo_arm_map_file_name = 'average_prob_fail_placebo_arm_map'
    average_prob_fail_drug_arm_map_file_name    = 'average_prob_fail_drug_arm_map'
    folder                                      = '/Users/juanromero/Documents/Python_3_Files/useless_folder'

    [average_log_hazard_ratio_map, average_prob_fail_placebo_arm_map, average_prob_fail_drug_arm_map] = \
        generate_maps(monthly_mean_min,
                      monthly_mean_max,
                      monthly_std_dev_min,
                      monthly_std_dev_max,
                      num_baseline_months_per_patient,
                      num_testing_months_per_patient,
                      min_req_base_sz_count,
                      num_theo_patients_per_trial_arm,
                      placebo_mu,
                      placebo_sigma,
                      drug_mu,
                      drug_sigma,
                      num_trials)

    if( not os.path.isdir(folder) ):
        os.makedirs(folder)

    with open(folder + '/' + average_log_hazard_ratio_map_file_name, 'w+') as json_file:
        json.dump(average_log_hazard_ratio_map.tolist(), json_file)
    
    with open(folder + '/' + average_prob_fail_placebo_arm_map_file_name, 'w+') as json_file:
        json.dump(average_prob_fail_placebo_arm_map.tolist(), json_file)

    with open(folder + '/' + average_prob_fail_drug_arm_map_file_name, 'w+') as json_file:
        json.dump(average_prob_fail_drug_arm_map.tolist(), json_file)

    print('\n' + pd.DataFrame(average_log_hazard_ratio_map).to_string()          + '\n\n' + \
                 pd.DataFrame(100*average_prob_fail_placebo_arm_map).to_string() + '\n\n' + \
                 pd.DataFrame(100*average_prob_fail_drug_arm_map).to_string()    + '\n' )
'''
    
