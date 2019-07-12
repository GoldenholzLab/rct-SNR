import numpy as np
import scipy.stats as stats
from lifelines.statistics import logrank_test
import time
import os


def generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                        monthly_std_dev_min, monthly_std_dev_max, 
                        num_patients_per_trial_arm):

    patient_pop_monthly_params = np.zeros((num_patients_per_trial_arm, 4))

    for patient_index in range(num_patients_per_trial_arm):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        daily_mean = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)
        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)

        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        patient_pop_monthly_params[patient_index, 0] = monthly_mean
        patient_pop_monthly_params[patient_index, 1] = monthly_std_dev
        patient_pop_monthly_params[patient_index, 2] = daily_n
        patient_pop_monthly_params[patient_index, 3] = daily_odds_ratio
    
    return patient_pop_monthly_params


def generate_daily_patient_diaries(patient_pop_params, num_patients_per_trial_arm, min_req_base_sz_count,
                                   num_days_per_patient_baseline, num_days_per_patient_total):

    daily_patient_diaries = np.zeros((num_patients_per_trial_arm, num_days_per_patient_total))

    for patient_index in range(num_patients_per_trial_arm):

        daily_n          = patient_pop_params[patient_index, 2]
        daily_odds_ratio = patient_pop_params[patient_index, 3]

        acceptable_baseline = False
        current_daily_patient_diary = np.zeros(num_days_per_patient_total)

        while(not acceptable_baseline):

            for day_index in range(num_days_per_patient_total):

                daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
                daily_count = np.random.poisson(daily_rate)

                current_daily_patient_diary[day_index] = daily_count
                
            current_patient_baseline_sz_count = np.sum(current_daily_patient_diary[:num_days_per_patient_baseline])
                
            if(current_patient_baseline_sz_count >= min_req_base_sz_count):

                    acceptable_baseline = True
            
        daily_patient_diaries[patient_index, :] = current_daily_patient_diary
    
    return daily_patient_diaries


def count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies, testing_daily_seizure_diaries, 
                                           num_days_per_patient_testing,         num_patients_per_trial_arm):

    TTP_times = np.zeros(num_patients_per_trial_arm)

    for patient_index in range(num_patients_per_trial_arm):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries[patient_index, day_index]

            reached_count = ( ( sum_count >= baseline_monthly_seizure_frequencies[patient_index] )  or ( day_index == (num_days_per_patient_testing - 1) ) )

            day_index = day_index + 1
    
        TTP_times[patient_index] = day_index

    return TTP_times


def calculate_individual_patient_endpoints(daily_patient_diaries,         num_patients_per_trial_arm, 
                                           num_days_per_patient_baseline, num_days_per_patient_testing):
    
    baseline_daily_seizure_diaries = daily_patient_diaries[:, :num_days_per_patient_baseline]
    testing_daily_seizure_diaries  = daily_patient_diaries[:, num_days_per_patient_baseline:]
    
    baseline_daily_seizure_frequencies = np.mean(baseline_daily_seizure_diaries, 1)
    testing_daily_seizure_frequencies = np.mean(testing_daily_seizure_diaries, 1)

    baseline_monthly_seizure_frequencies = baseline_daily_seizure_frequencies*28

    for patient_index in range(num_patients_per_trial_arm):
        if(baseline_daily_seizure_frequencies[patient_index] == 0):
            baseline_daily_seizure_frequencies[patient_index] = 0.000000001
    
    percent_changes = np.divide(baseline_daily_seizure_frequencies - testing_daily_seizure_frequencies, baseline_daily_seizure_frequencies)

    TTP_times = count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies, testing_daily_seizure_diaries, 
                                           num_days_per_patient_testing,         num_patients_per_trial_arm)

    return [percent_changes, TTP_times]


def apply_effect(daily_patient_diaries,         num_patients_per_trial_arm, 
                 num_days_per_patient_baseline, num_days_per_patient_testing, 
                 effect_mu,                     effect_sigma):

    testing_daily_patient_diaries = daily_patient_diaries[:, num_days_per_patient_baseline:]

    for patient_index in range(num_patients_per_trial_arm):

        effect = np.random.normal(effect_mu, effect_sigma)
        if(effect > 1):
            effect = 1

        current_testing_daily_patient_diary = testing_daily_patient_diaries[patient_index, :]

        for day_index in range(num_days_per_patient_testing):

            current_seizure_count = current_testing_daily_patient_diary[day_index]
            num_removed = 0

            for seizure_index in range(np.int_(current_seizure_count)):

                if(np.random.random() <= np.abs(effect)):

                    num_removed = num_removed + np.sign(effect)
            
            current_seizure_count = current_seizure_count - num_removed
            current_testing_daily_patient_diary[day_index] = current_seizure_count

        testing_daily_patient_diaries[patient_index, :] = current_testing_daily_patient_diary
    
    daily_patient_diaries[:, num_days_per_patient_baseline:] = testing_daily_patient_diaries

    return daily_patient_diaries


def generate_trial_outcomes(patient_pop_placebo_arm_params, patient_pop_drug_arm_params, 
                            min_req_base_sz_count, num_patients_per_trial_arm,
                            num_days_per_patient_baseline, num_days_per_patient_testing, num_days_per_patient_total,
                            placebo_mu, placebo_sigma, drug_mu, drug_sigma):

    placebo_arm_daily_patient_diaries = \
            generate_daily_patient_diaries(patient_pop_placebo_arm_params, num_patients_per_trial_arm, min_req_base_sz_count,
                                           num_days_per_patient_baseline, num_days_per_patient_total)

    placebo_arm_daily_patient_diaries = \
        apply_effect(placebo_arm_daily_patient_diaries, num_patients_per_trial_arm, 
                     num_days_per_patient_baseline, num_days_per_patient_testing,
                     placebo_mu, placebo_sigma)

    drug_arm_daily_patient_diaries = \
        generate_daily_patient_diaries(patient_pop_drug_arm_params, num_patients_per_trial_arm, min_req_base_sz_count,
                                       num_days_per_patient_baseline, num_days_per_patient_total)
        
    drug_arm_daily_patient_diaries = \
        apply_effect(drug_arm_daily_patient_diaries, num_patients_per_trial_arm, 
                     num_days_per_patient_baseline, num_days_per_patient_testing,
                     placebo_mu, placebo_sigma)
        
    drug_arm_daily_patient_diaries = \
        apply_effect(drug_arm_daily_patient_diaries, num_patients_per_trial_arm, 
                     num_days_per_patient_baseline, num_days_per_patient_testing,
                     drug_mu, drug_sigma)
        
    [placebo_arm_percent_changes, placebo_arm_TTP_times] = \
        calculate_individual_patient_endpoints(placebo_arm_daily_patient_diaries, num_patients_per_trial_arm, 
                                               num_days_per_patient_baseline,     num_days_per_patient_testing)
        
    [drug_arm_percent_changes, drug_arm_TTP_times] = \
        calculate_individual_patient_endpoints(drug_arm_daily_patient_diaries, num_patients_per_trial_arm, 
                                               num_days_per_patient_baseline,     num_days_per_patient_testing)

    num_placebo_50_percent_responders     = np.sum(placebo_arm_percent_changes > 0.5)
    num_placebo_50_percent_non_responders = num_patients_per_trial_arm - num_placebo_50_percent_responders
    num_drug_50_percent_responders        = np.sum(drug_arm_percent_changes > 0.5)
    num_drug_50_percent_non_responders    = num_patients_per_trial_arm - num_drug_50_percent_responders
    table = np.array([[num_placebo_50_percent_responders, num_placebo_50_percent_non_responders], [num_drug_50_percent_responders, num_drug_50_percent_non_responders]])

    events_observed_placebo = np.ones(len(placebo_arm_TTP_times))
    events_observed_drug = np.ones(len(drug_arm_TTP_times))
    TTP_results = logrank_test(placebo_arm_TTP_times, drug_arm_TTP_times, events_observed_placebo, events_observed_drug)

    placebo_arm_RR50 = 100*num_placebo_50_percent_responders/num_patients_per_trial_arm
    placebo_arm_MPC = 100*np.median(placebo_arm_percent_changes)
    placebo_arm_TTP = np.median(placebo_arm_TTP_times)

    drug_arm_RR50 = 100*num_drug_50_percent_responders/num_patients_per_trial_arm
    drug_arm_MPC = 100*np.median(drug_arm_percent_changes)
    drug_arm_TTP = np.median(drug_arm_TTP_times)

    [_, RR50_p_value] = stats.fisher_exact(table)
    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)
    TTP_p_value = TTP_results.p_value

    return [placebo_arm_RR50, placebo_arm_MPC, placebo_arm_TTP, 
            drug_arm_RR50,    drug_arm_MPC,    drug_arm_TTP,
            RR50_p_value,     MPC_p_value,     TTP_p_value]


def estimate_endpoint_statistics(patient_pop_placebo_arm_params, patient_pop_drug_arm_params,
                                 num_months_per_patient_baseline, num_months_per_patient_testing, 
                                 min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
                                 placebo_mu, placebo_sigma, drug_mu, drug_sigma):

    num_days_per_patient_baseline = num_months_per_patient_baseline*28
    num_days_per_patient_testing = num_months_per_patient_testing*28
    num_days_per_patient_total = num_days_per_patient_baseline + num_days_per_patient_testing

    placebo_arm_RR50_array = np.zeros(num_trials)
    placebo_arm_MPC_array  = np.zeros(num_trials)
    placebo_arm_TTP_array  = np.zeros(num_trials)
    drug_arm_RR50_array    = np.zeros(num_trials)
    drug_arm_MPC_array     = np.zeros(num_trials)
    drug_arm_TTP_array     = np.zeros(num_trials)
    RR50_p_value_array     = np.zeros(num_trials)
    MPC_p_value_array      = np.zeros(num_trials)
    TTP_p_value_array      = np.zeros(num_trials)

    for trial_index in range(num_trials):

        trial_start_time_in_seconds = time.time()

        [placebo_arm_RR50, placebo_arm_MPC, placebo_arm_TTP, 
         drug_arm_RR50,    drug_arm_MPC,    drug_arm_TTP,
         RR50_p_value,     MPC_p_value,     TTP_p_value  ] = \
            generate_trial_outcomes(patient_pop_placebo_arm_params, patient_pop_drug_arm_params, 
                                    min_req_base_sz_count, num_patients_per_trial_arm,
                                    num_days_per_patient_baseline, num_days_per_patient_testing, num_days_per_patient_total,
                                    placebo_mu, placebo_sigma, drug_mu, drug_sigma)

        placebo_arm_RR50_array[trial_index] = placebo_arm_RR50
        placebo_arm_MPC_array[trial_index]  = placebo_arm_MPC
        placebo_arm_TTP_array[trial_index]  = placebo_arm_TTP
        drug_arm_RR50_array[trial_index]    = drug_arm_RR50
        drug_arm_MPC_array[trial_index]     = drug_arm_MPC
        drug_arm_TTP_array[trial_index]     = drug_arm_TTP
        RR50_p_value_array[trial_index]     = RR50_p_value
        MPC_p_value_array[trial_index]      = MPC_p_value
        TTP_p_value_array[trial_index]      = TTP_p_value

        trial_stop_time_in_seconds = time.time()
        trial_runtime_in_seconds_str = 'trial #' + str(trial_index + 1) + ' runtime: ' + str(np.round(trial_stop_time_in_seconds - trial_start_time_in_seconds, 3)) + ' seconds'
        print(trial_runtime_in_seconds_str)

    expected_placebo_arm_RR50 = np.mean(placebo_arm_RR50_array)
    expected_placebo_arm_MPC  = np.mean(placebo_arm_MPC_array)
    expected_placebo_arm_TTP  = np.mean(placebo_arm_TTP_array)
    expected_drug_arm_RR50    = np.mean(drug_arm_RR50_array)
    expected_drug_arm_MPC     = np.mean(drug_arm_MPC_array)
    expected_drug_arm_TTP     = np.mean(drug_arm_TTP_array)
    RR50_stat_power           = 100*np.sum(RR50_p_value_array < 0.05)/num_trials
    MPC_stat_power            = 100*np.sum(MPC_p_value_array < 0.05)/num_trials
    TTP_stat_power            = 100*np.sum(TTP_p_value_array < 0.05)/num_trials

    return [expected_placebo_arm_RR50, expected_placebo_arm_MPC, expected_placebo_arm_TTP,
            expected_drug_arm_RR50,    expected_drug_arm_MPC,    expected_drug_arm_TTP,
            RR50_stat_power,           MPC_stat_power,           TTP_stat_power]


if(__name__ == '__main__'):
    
    monthly_mean_min = 1
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16
    min_req_base_sz_count = 4

    num_patients_per_trial_arm = 153
    num_months_per_patient_baseline = 2
    num_months_per_patient_testing = 3
    num_trials = 5000

    placebo_mu = 0
    placebo_sigma  = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05

    data_file_name = 'random_population_endpoint_statistics'

    start_time_in_seconds = time.time()

    patient_pop_placebo_arm_params = \
        generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                            monthly_std_dev_min, monthly_std_dev_max, 
                            num_patients_per_trial_arm)
    
    patient_pop_drug_arm_params = \
        generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                            monthly_std_dev_min, monthly_std_dev_max, 
                            num_patients_per_trial_arm)

    [expected_placebo_arm_RR50, expected_placebo_arm_MPC, expected_placebo_arm_TTP,
     expected_drug_arm_RR50,    expected_drug_arm_MPC,    expected_drug_arm_TTP,
     RR50_stat_power,           MPC_stat_power,           TTP_stat_power            ] = \
         estimate_endpoint_statistics(patient_pop_placebo_arm_params, patient_pop_drug_arm_params,
                                      num_months_per_patient_baseline, num_months_per_patient_testing, 
                                      min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
                                      placebo_mu, placebo_sigma, drug_mu, drug_sigma)

    stop_time_in_seconds = time.time()
    total_time_in_minutes = (stop_time_in_seconds - start_time_in_seconds)/60

    data_str = '\n\n' + 'expected placebo arm 50% responder rate:              ' + str(np.round(expected_placebo_arm_RR50, 3)) + ' %\n' + \
                        'expected drug arm 50% responder rate:                 ' + str(np.round(expected_drug_arm_RR50, 3))    + ' %\n' + \
                        '50% responder rate empirical statistical power:       ' + str(np.round(RR50_stat_power, 3))           + ' %\n' + \
                        'expected placebo arm median percent change:           ' + str(np.round(expected_placebo_arm_MPC, 3))  + ' %\n' + \
                        'expected drug arm median percent change:              ' + str(np.round(expected_drug_arm_MPC, 3))     + ' %\n' + \
                        'median percent change empirical statistical power:    ' + str(np.round(MPC_stat_power, 3))            + ' %\n' + \
                        'expected placebo arm time-to-prerandomization:        ' + str(np.round(expected_placebo_arm_TTP, 3))  +   '\n' + \
                        'expected drug arm time-to-prerandomization:           ' + str(np.round(expected_drug_arm_TTP, 3))     +   '\n' + \
                        'time-to-prerandomization empirical statistical power: ' + str(np.round(TTP_stat_power, 3))            + ' %\n' + \
                        'total runtime:                                        ' + str(np.round(total_time_in_minutes, 3))     + ' minutes'

    data_file_path = os.getcwd() + '/' + data_file_name + '.txt'

    with open(data_file_path, 'w+') as text_file:

        text_file.write(data_str)
    
    

