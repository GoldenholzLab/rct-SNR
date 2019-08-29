import numpy as np
import os
import json
import subprocess
import pandas as pd
from lifelines.statistics import logrank_test
import time


def generate_patient_pop_params(monthly_mean_min,
                                monthly_mean_max, 
                                monthly_std_dev_min, 
                                monthly_std_dev_max, 
                                num_theo_patients_per_trial_arm):

    patient_pop_monthly_param_sets = np.zeros((num_theo_patients_per_trial_arm , 2))

    for theo_patient_index in range(num_theo_patients_per_trial_arm ):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        patient_pop_monthly_param_sets[theo_patient_index, 0] = monthly_mean
        patient_pop_monthly_param_sets[theo_patient_index, 1] = monthly_std_dev

    return patient_pop_monthly_param_sets


def retrieve_map_location_TTP_times(folder, monthly_mean, monthly_std_dev, trial_arm_str):

    file_name = str(monthly_mean) + '_' + str(monthly_std_dev) + '_' + trial_arm_str
    file_path = folder + '/' + file_name + '.json'
    with open(file_path, 'r') as json_file:
        TTP_times = np.array(json.load(json_file))

    return TTP_times


def retrieve_theo_patient_pop_TTP_times(patient_pop_monthly_param_set, 
                                        num_theo_patients_per_trial_arm,
                                        folder,
                                        trial_arm_str):
    
    trial_arm_TTP_times = np.array([])

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = int(patient_pop_monthly_param_set[theo_patient_index, 0])
        monthly_std_dev = int(patient_pop_monthly_param_set[theo_patient_index, 1])

        tmp_trial_arm_TTP_times = retrieve_map_location_TTP_times(folder, monthly_mean, monthly_std_dev, trial_arm_str)

        trial_arm_TTP_times = np.concatenate([trial_arm_TTP_times, tmp_trial_arm_TTP_times])
    
    num_TTP_times = len(trial_arm_TTP_times)

    trial_arm_TTP_observed_array = np.ones(num_TTP_times)
    trial_arm_TTP_observed_array[ trial_arm_TTP_times == 84 ] = 0

    return [trial_arm_TTP_times, trial_arm_TTP_observed_array]


def calculate_one_trial_analytical_quantities(placebo_arm_TTP_times,
                                              placebo_arm_TTP_observed_array,
                                              drug_arm_TTP_times,
                                              drug_arm_TTP_observed_array,
                                              num_theo_patients_per_trial_arm,
                                              num_patients_per_dot,
                                              tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_arm_TTP_times, drug_arm_TTP_times)
    events                 = np.append(placebo_arm_TTP_observed_array, drug_arm_TTP_observed_array)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*num_patients_per_dot*['C']), np.array(num_theo_patients_per_trial_arm*num_patients_per_dot*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'estimate_log_hazard_ratio.R', relative_tmp_file_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])

    os.remove(relative_tmp_file_path)

    prob_fail_placebo_arm = np.sum(placebo_arm_TTP_observed_array == True)/(num_theo_patients_per_trial_arm*num_patients_per_dot)
    prob_fail_drug_arm    = np.sum(drug_arm_TTP_observed_array == True)/(num_theo_patients_per_trial_arm*num_patients_per_dot)

    return [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm]


def estimate_map_based_statistical_power(placebo_arm_patient_pop_monthly_param_set,
                                         drug_arm_patient_pop_monthly_param_set,
                                         num_theo_patients_per_trial_arm,
                                         num_patients_per_dot,
                                         alpha,
                                         folder):

    placebo_arm_str = 'placebo'
    drug_arm_str    = 'drug'
    tmp_file_name   = 'tmp'

    [placebo_arm_TTP_times, 
     placebo_arm_TTP_observed_array] = \
        retrieve_theo_patient_pop_TTP_times(placebo_arm_patient_pop_monthly_param_set, 
                                            num_theo_patients_per_trial_arm,
                                            folder,
                                            placebo_arm_str)
    
    [drug_arm_TTP_times, 
     drug_arm_TTP_observed_array] = \
        retrieve_theo_patient_pop_TTP_times(drug_arm_patient_pop_monthly_param_set, 
                                            num_theo_patients_per_trial_arm,
                                            folder,
                                            drug_arm_str)

    [postulated_log_hazard_ratio, 
     prob_fail_placebo_arm, 
     prob_fail_drug_arm         ] = \
        calculate_one_trial_analytical_quantities(placebo_arm_TTP_times,
                                                  placebo_arm_TTP_observed_array,
                                                  drug_arm_TTP_times,
                                                  drug_arm_TTP_observed_array,
                                                  num_theo_patients_per_trial_arm,
                                                  num_patients_per_dot,
                                                  tmp_file_name)

    command        = ['Rscript', 'calculate_cph_power.R', str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm), 
                      str(prob_fail_drug_arm), str(prob_fail_placebo_arm), str(np.exp(postulated_log_hazard_ratio)), str(alpha)]
    process        = subprocess.Popen(command, stdout=subprocess.PIPE)
    map_stat_power = 100*float(process.communicate()[0].decode().split()[1])

    return map_stat_power


def generate_one_trial_arm_of_patient_diaries(patient_pop_monthly_param_sets, 
                                              num_patients_per_trial_arm, 
                                              min_req_base_sz_count,
                                              num_baseline_days_per_patient, 
                                              num_total_days_per_patient):

    daily_patient_diaries = np.zeros((num_patients_per_trial_arm, num_total_days_per_patient))

    for patient_index in range(num_patients_per_trial_arm):

        monthly_mean    = patient_pop_monthly_param_sets[patient_index, 0]
        monthly_std_dev = patient_pop_monthly_param_sets[patient_index, 1]

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


def generate_individual_patient_TTP_times_per_trial_arm(patient_pop_monthly_param_sets, 
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
        generate_one_trial_arm_of_patient_diaries(patient_pop_monthly_param_sets, 
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


def generate_one_trial_TTP_times(placebo_arm_patient_pop_monthly_param_sets, 
                                 drug_arm_patient_pop_monthly_param_sets,
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
        generate_individual_patient_TTP_times_per_trial_arm(placebo_arm_patient_pop_monthly_param_sets, 
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
        generate_individual_patient_TTP_times_per_trial_arm(drug_arm_patient_pop_monthly_param_sets, 
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


def calculate_one_trial_p_value(placebo_arm_patient_pop_monthly_param_sets,
                                drug_arm_patient_pop_monthly_param_sets,
                                num_theo_patients_per_trial_arm,
                                num_baseline_days_per_patient,
                                num_testing_days_per_patient,
                                num_total_days_per_patient,
                                min_req_base_sz_count,
                                placebo_mu,
                                placebo_sigma,
                                drug_mu,
                                drug_sigma,
                                tmp_file_name):

    [one_placebo_arm_TTP_times, one_placebo_arm_observed_array, 
     one_drug_arm_TTP_times,    one_drug_arm_observed_array   ] = \
        generate_one_trial_TTP_times(placebo_arm_patient_pop_monthly_param_sets, 
                                     drug_arm_patient_pop_monthly_param_sets,
                                     min_req_base_sz_count,
                                     num_baseline_days_per_patient,
                                     num_testing_days_per_patient,
                                     num_total_days_per_patient,
                                     num_theo_patients_per_trial_arm,
                                     placebo_mu,
                                     placebo_sigma,
                                     drug_mu,
                                     drug_sigma)

    TTP_results = logrank_test(one_placebo_arm_TTP_times,
                               one_drug_arm_TTP_times, 
                               one_placebo_arm_observed_array, 
                               one_drug_arm_observed_array)
    TTP_p_value = TTP_results.p_value

    [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm] = \
        calculate_one_trial_analytical_quantities(one_placebo_arm_TTP_times,
                                                  one_placebo_arm_observed_array,
                                                  one_drug_arm_TTP_times,
                                                  one_drug_arm_observed_array,
                                                  num_theo_patients_per_trial_arm,
                                                  1,
                                                  tmp_file_name)

    return TTP_p_value


def calculate_empirical_and_semianalytical_statistical_power(placebo_arm_patient_pop_monthly_param_sets,
                                                             drug_arm_patient_pop_monthly_param_sets,
                                                             num_theo_patients_per_trial_arm,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_total_days_per_patient,
                                                             min_req_base_sz_count,
                                                             placebo_mu,
                                                             placebo_sigma,
                                                             drug_mu,
                                                             drug_sigma,
                                                             num_trials,
                                                             stat_power_estimate_index):

    p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        trial_start_time_in_seconds = time.time()

        tmp_file_name = str(trial_index) + '_' + str(stat_power_estimate_index)

        p_value_array[trial_index] = \
            calculate_one_trial_p_value(placebo_arm_patient_pop_monthly_param_sets,
                                        drug_arm_patient_pop_monthly_param_sets,
                                        num_theo_patients_per_trial_arm,
                                        num_baseline_days_per_patient,
                                        num_testing_days_per_patient,
                                        num_total_days_per_patient,
                                        min_req_base_sz_count,
                                        placebo_mu,
                                        placebo_sigma,
                                        drug_mu,
                                        drug_sigma,
                                        tmp_file_name)
        
        trial_stop_time_in_seconds = time.time()
        trial_runtime_str = str(np.round((trial_stop_time_in_seconds - trial_start_time_in_seconds)/60, 3))
        print( 'trial #' + str(trial_index + 1) + ', runtime: ' + trial_runtime_str + ' minutes' )
    
    emp_stat_power = 100*np.sum(p_value_array < 0.05)/num_trials

    return emp_stat_power


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    min_req_base_sz_count           = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials                      = 100
    num_theo_patients_per_trial_arm = 153
    num_patients_per_dot            = 5000
    alpha                           = 0.05

    hist_maps_folder = os.getcwd() + '/hist_maps_folder'

    #-----------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    placebo_arm_patient_pop_monthly_param_set = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)
    
    drug_arm_patient_pop_monthly_param_set = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)

    [emp_stat_power, semi_ana_stat_power] = \
        calculate_empirical_and_semianalytical_statistical_power(placebo_arm_patient_pop_monthly_param_set,
                                                                 drug_arm_patient_pop_monthly_param_set,
                                                                 num_theo_patients_per_trial_arm,
                                                                 num_baseline_days_per_patient,
                                                                 num_testing_days_per_patient,
                                                                 num_total_days_per_patient,
                                                                 min_req_base_sz_count,
                                                                 placebo_mu,
                                                                 placebo_sigma,
                                                                 drug_mu,
                                                                 drug_sigma,
                                                                 num_trials,
                                                                 0)

    map_stat_power = \
        estimate_map_based_statistical_power(placebo_arm_patient_pop_monthly_param_set,
                                             drug_arm_patient_pop_monthly_param_set, 
                                             num_theo_patients_per_trial_arm,
                                             num_patients_per_dot,
                                             alpha,
                                             hist_maps_folder)

    print([emp_stat_power, semi_ana_stat_power, map_stat_power])

    #-----------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#

    