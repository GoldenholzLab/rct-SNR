import numpy as np
from lifelines.statistics import logrank_test
import scipy.stats as stats
import subprocess
import time


def generate_patient_pop_params(monthly_mean_min,
                                monthly_mean_max, 
                                monthly_std_dev_min, 
                                monthly_std_dev_max, 
                                num_theo_patients_per_trial_arm ):

    patient_monthly_params = np.zeros((num_theo_patients_per_trial_arm , 2))
    patient_daily_params = np.zeros((num_theo_patients_per_trial_arm , 2))

    for theo_patient_index in range(num_theo_patients_per_trial_arm ):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        daily_mean    = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)

        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)
        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        patient_monthly_params[theo_patient_index, 0] = monthly_mean
        patient_monthly_params[theo_patient_index, 1] = monthly_std_dev

        patient_daily_params[theo_patient_index, 0] = daily_n
        patient_daily_params[theo_patient_index, 1] = daily_odds_ratio

    return [patient_monthly_params, patient_daily_params]


def generate_daily_seizure_diary(daily_n, 
                                 daily_odds_ratio,
                                 num_baseline_days_per_patient,
                                 num_total_days_per_patient,
                                 min_req_bs_sz_count):
    
    seizure_diary = np.zeros(num_total_days_per_patient)
    acceptable_baseline = False

    while (not acceptable_baseline):

        for day_index in range(num_total_days_per_patient):

            daily_rate  = np.random.gamma(daily_n, daily_odds_ratio)
            daily_count = np.random.poisson(daily_rate)

            seizure_diary[day_index] = daily_count
    
        baseline_sz_count = np.sum(seizure_diary[:num_baseline_days_per_patient])

        if(baseline_sz_count >= min_req_bs_sz_count):

            acceptable_baseline = True
    
    return seizure_diary


def generate_one_trial_arm_of_seizure_diaries(patient_daily_params, 
                                              num_theo_patients_per_trial_arm,
                                              num_baseline_days_per_patient,
                                              num_total_days_per_patient,
                                              min_req_bs_sz_count):

    one_trial_arm_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_total_days_per_patient))

    for patient_index in range(num_theo_patients_per_trial_arm):

        daily_n          = patient_daily_params[patient_index, 0]
        daily_odds_ratio = patient_daily_params[patient_index, 1]

        one_trial_arm_seizure_diaries[patient_index, :] = \
            generate_daily_seizure_diary(daily_n, 
                                         daily_odds_ratio,
                                         num_baseline_days_per_patient,
                                         num_total_days_per_patient,
                                         min_req_bs_sz_count)
    
    return one_trial_arm_seizure_diaries


def generate_one_map_point_of_seizure_diaries(daily_n, 
                                              daily_odds_ratio,
                                              num_patients_per_monthly_param_pair,
                                              num_baseline_days_per_patient,
                                              num_total_days_per_patient,
                                              min_req_bs_sz_count):

    one_map_point_seizure_diaries = np.zeros((num_patients_per_monthly_param_pair, num_total_days_per_patient))

    for patient_index in range(num_patients_per_monthly_param_pair):

        one_map_point_seizure_diaries[patient_index, :] = \
            generate_daily_seizure_diary(daily_n, 
                                         daily_odds_ratio,
                                         num_baseline_days_per_patient,
                                         num_total_days_per_patient,
                                         min_req_bs_sz_count)

    return one_map_point_seizure_diaries


def calculate_individual_patient_endpoints_per_diary_set(daily_seizure_diaries,
                                                         num_baseline_days_per_patient,
                                                         num_testing_days_per_patient,
                                                         num_daily_seizure_diaries):

    baseline_daily_seizure_diaries = daily_seizure_diaries[:, :num_baseline_days_per_patient]
    testing_daily_seizure_diaries  = daily_seizure_diaries[:, num_baseline_days_per_patient:]

    baseline_daily_seizure_frequencies = np.mean(baseline_daily_seizure_diaries, 1)
    testing_daily_seizure_frequencies  = np.mean(testing_daily_seizure_diaries, 1)

    for daily_seizure_diary_index in range(num_daily_seizure_diaries):
        if( np.sum( baseline_daily_seizure_frequencies[daily_seizure_diary_index] ) == 0 ):
            baseline_daily_seizure_frequencies[daily_seizure_diary_index] = 0.0000001
    
    percent_changes = np.divide(baseline_daily_seizure_frequencies - testing_daily_seizure_frequencies, baseline_daily_seizure_frequencies)
    TTP_times = np.zeros(num_daily_seizure_diaries)

    baseline_monthly_seizure_frequencies = 28*baseline_daily_seizure_frequencies

    for daily_seizure_diary_index in range(num_daily_seizure_diaries):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries[daily_seizure_diary_index, day_index]
            reached_count = ( ( sum_count >= baseline_monthly_seizure_frequencies[daily_seizure_diary_index] )  or ( day_index == (num_testing_days_per_patient - 1) ) )
            day_index = day_index + 1
        
        TTP_times[daily_seizure_diary_index] = day_index
    
    return [percent_changes, TTP_times]


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


def generate_individual_patient_endpoints_per_trial(patient_pop_daily_params, 
                                                    num_theo_patients_per_trial_arm,
                                                    num_baseline_days_per_patient,
                                                    num_testing_days_per_patient,
                                                    num_total_days_per_patient,
                                                    min_req_bs_sz_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma):

    one_trial_arm_daily_seizure_diaries = \
        generate_one_trial_arm_of_seizure_diaries(patient_pop_daily_params, 
                                                  num_theo_patients_per_trial_arm,
                                                  num_baseline_days_per_patient,
                                                  num_total_days_per_patient,
                                                  min_req_bs_sz_count)
        
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

    [one_trial_arm_percent_changes, one_trial_arm_TTP_times] = \
        calculate_individual_patient_endpoints_per_diary_set(one_trial_arm_daily_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_theo_patients_per_trial_arm)
    
    return [one_trial_arm_percent_changes, one_trial_arm_TTP_times]


def generate_individual_patient_endpoints_per_map_point(placebo_arm_daily_n,
                                                        placebo_arm_daily_odds_ratio,
                                                        drug_arm_daily_n, 
                                                        drug_arm_daily_odds_ratio,
                                                        num_baseline_days_per_patient,
                                                        num_testing_days_per_patient,
                                                        num_total_days_per_patient,
                                                        min_req_bs_sz_count,
                                                        num_trials,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma):

    one_placebo_map_point_seizure_diaries = \
        generate_one_map_point_of_seizure_diaries(placebo_arm_daily_n, 
                                                  placebo_arm_daily_odds_ratio, 
                                                  num_trials,
                                                  num_baseline_days_per_patient,
                                                  num_total_days_per_patient,
                                                  min_req_bs_sz_count)

    one_placebo_map_point_seizure_diaries = \
        apply_effect(one_placebo_map_point_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_trials,
                     placebo_mu,
                     placebo_sigma)
    
    [one_placebo_map_point_percent_changes, one_placebo_map_point_TTP_times] = \
        calculate_individual_patient_endpoints_per_diary_set(one_placebo_map_point_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_trials)

    one_drug_map_point_seizure_diaries = \
        generate_one_map_point_of_seizure_diaries(drug_arm_daily_n, 
                                                  drug_arm_daily_odds_ratio, 
                                                  num_trials,
                                                  num_baseline_days_per_patient,
                                                  num_total_days_per_patient,
                                                  min_req_bs_sz_count)
    
    one_drug_map_point_seizure_diaries = \
        apply_effect(one_placebo_map_point_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_trials,
                     placebo_mu,
                     placebo_sigma)
    
    one_drug_map_point_seizure_diaries = \
        apply_effect(one_placebo_map_point_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_trials,
                     drug_mu,
                     drug_sigma)
    
    [one_drug_map_point_percent_changes, one_drug_map_point_TTP_times] = \
        calculate_individual_patient_endpoints_per_diary_set(one_drug_map_point_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_trials)
    
    return [one_placebo_map_point_percent_changes, one_placebo_map_point_TTP_times, 
            one_drug_map_point_percent_changes,    one_drug_map_point_TTP_times    ]


def generate_p_values_per_trial(patient_drug_pop_daily_params,
                                patient_placebo_pop_daily_params, 
                                num_theo_patients_per_trial_arm,
                                num_baseline_days_per_patient,
                                num_testing_days_per_patient,
                                num_total_days_per_patient,
                                min_req_bs_sz_count,
                                placebo_mu,
                                placebo_sigma,
                                drug_mu,
                                drug_sigma):

    [one_placebo_arm_percent_changes, one_placebo_arm_TTP_times] = \
        generate_individual_patient_endpoints_per_trial(patient_placebo_pop_daily_params, 
                                                        num_theo_patients_per_trial_arm,
                                                        num_baseline_days_per_patient,
                                                        num_testing_days_per_patient,
                                                        num_total_days_per_patient,
                                                        min_req_bs_sz_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        0,
                                                        0)

    [one_drug_arm_percent_changes, one_drug_arm_TTP_times] = \
        generate_individual_patient_endpoints_per_trial(patient_drug_pop_daily_params, 
                                                        num_theo_patients_per_trial_arm,
                                                        num_baseline_days_per_patient,
                                                        num_testing_days_per_patient,
                                                        num_total_days_per_patient,
                                                        min_req_bs_sz_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma)
        
    num_placebo_50_percent_responders     = np.sum(one_placebo_arm_percent_changes >= 0.5)
    num_placebo_50_percent_non_responders = num_theo_patients_per_trial_arm - num_placebo_50_percent_responders
    num_drug_50_percent_responders        = np.sum(one_drug_arm_percent_changes >= 0.5)
    num_drug_50_percent_non_responders    = num_theo_patients_per_trial_arm - num_drug_50_percent_responders
    table = np.array([[num_placebo_50_percent_responders, num_placebo_50_percent_non_responders], [num_drug_50_percent_responders, num_drug_50_percent_non_responders]])

    placebo_arm_RR50 = 100*num_placebo_50_percent_responders/num_theo_patients_per_trial_arm
    drug_arm_RR50    = 100*num_drug_50_percent_responders/num_theo_patients_per_trial_arm

    events_observed_placebo = np.ones(len(one_placebo_arm_TTP_times))
    events_observed_drug    = np.ones(len(one_drug_arm_TTP_times))
    TTP_results = logrank_test(one_placebo_arm_TTP_times, one_drug_arm_TTP_times, events_observed_placebo, events_observed_drug)

    [_, RR50_p_value] = stats.fisher_exact(table)
    [_, MPC_p_value]  = stats.ranksums(one_placebo_arm_percent_changes, one_drug_arm_percent_changes)
    TTP_p_value = TTP_results.p_value

    return [placebo_arm_RR50, drug_arm_RR50, RR50_p_value, MPC_p_value, TTP_p_value]


def calculate_empirical_stat_power(patient_drug_pop_daily_params,
                                   patient_placebo_pop_daily_params, 
                                   num_theo_patients_per_trial_arm,
                                   num_baseline_days_per_patient,
                                   num_testing_days_per_patient,
                                   num_total_days_per_patient,
                                   min_req_bs_sz_count,
                                   placebo_mu,
                                   placebo_sigma,
                                   drug_mu,
                                   drug_sigma,
                                   num_trials):

    placebo_arm_RR50_array = np.zeros(num_trials)
    drug_arm_RR50_array    = np.zeros(num_trials)
    RR50_p_value_array = np.zeros(num_trials)
    MPC_p_value_array  = np.zeros(num_trials)
    TTP_p_value_array  = np.zeros(num_trials)

    for trial_index in range(num_trials):

        trial_start_time_in_seconds = time.time()

        [placebo_arm_RR50, drug_arm_RR50, RR50_p_value, MPC_p_value, TTP_p_value] = \
            generate_p_values_per_trial(patient_drug_pop_daily_params,
                                        patient_placebo_pop_daily_params, 
                                        num_theo_patients_per_trial_arm,
                                        num_baseline_days_per_patient,
                                        num_testing_days_per_patient,
                                        num_total_days_per_patient,
                                        min_req_bs_sz_count,
                                        placebo_mu,
                                        placebo_sigma,
                                        drug_mu,
                                        drug_sigma)
        
        placebo_arm_RR50_array[trial_index] = placebo_arm_RR50
        drug_arm_RR50_array[trial_index]    = drug_arm_RR50

        RR50_p_value_array[trial_index] = RR50_p_value
        MPC_p_value_array[trial_index]  = MPC_p_value
        TTP_p_value_array[trial_index]  = TTP_p_value
        
        trial_stop_time_in_seconds      = time.time()
        total_trial_time_in_seconds     = trial_stop_time_in_seconds - trial_start_time_in_seconds
        total_trial_time_in_seconds_str = 'trial #' + str(trial_index + 1) + ' runtime: ' + str(np.round(total_trial_time_in_seconds, 3)) + ' seconds'
        print(total_trial_time_in_seconds_str)
    
    expected_emp_placebo_arm_RR50 = np.mean(placebo_arm_RR50_array)
    expected_emp_drug_arm_RR50    = np.mean(drug_arm_RR50_array)
    command = ['Rscript', 'Fisher_Exact_Power_Calc.R', str(expected_emp_placebo_arm_RR50), str(expected_emp_drug_arm_RR50), str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    fisher_exact_emp_stat_power = float(process.communicate()[0].decode().split()[1])

    RR50_emp_stat_power = 100*np.sum(RR50_p_value_array < 0.05)/num_trials
    MPC_emp_stat_power  = 100*np.sum(MPC_p_value_array < 0.05)/num_trials
    TTP_emp_stat_power  = 100*np.sum(TTP_p_value_array < 0.05)/num_trials

    return [expected_emp_placebo_arm_RR50, expected_emp_drug_arm_RR50, RR50_emp_stat_power, fisher_exact_emp_stat_power, MPC_emp_stat_power, TTP_emp_stat_power]


def calculate_analytical_stat_power(patient_placebo_pop_daily_params,
                                    patient_drug_pop_daily_params,
                                    num_theo_patients_per_trial_arm,
                                    num_baseline_days_per_patient,
                                    num_testing_days_per_patient,
                                    num_total_days_per_patient,
                                    min_req_bs_sz_count,
                                    num_trials,
                                    placebo_mu,
                                    placebo_sigma,
                                    drug_mu,
                                    drug_sigma):

    median_percent_change_per_placebo_arm_theo_patients = np.zeros(num_theo_patients_per_trial_arm)
    median_percent_change_per_drug_arm_theo_patients    = np.zeros(num_theo_patients_per_trial_arm)
    average_TTP_per_placebo_arm_theo_patients           = np.zeros(num_theo_patients_per_trial_arm)
    average_TTP_per_drug_arm_theo_patients              = np.zeros(num_theo_patients_per_trial_arm)

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        patient_point_start_time_in_seconds = time.time()

        placebo_arm_daily_n          = patient_placebo_pop_daily_params[theo_patient_index, 0]
        placebo_arm_daily_odds_ratio = patient_placebo_pop_daily_params[theo_patient_index, 1]
        drug_arm_daily_n             = patient_drug_pop_daily_params[theo_patient_index, 0]
        drug_arm_daily_odds_ratio    = patient_drug_pop_daily_params[theo_patient_index, 1]

        [one_placebo_map_point_percent_changes, one_placebo_map_point_TTP_times, 
         one_drug_map_point_percent_changes,    one_drug_map_point_TTP_times] = \
             generate_individual_patient_endpoints_per_map_point(placebo_arm_daily_n,
                                                                 placebo_arm_daily_odds_ratio,
                                                                 drug_arm_daily_n, 
                                                                 drug_arm_daily_odds_ratio,
                                                                 num_baseline_days_per_patient,
                                                                 num_testing_days_per_patient,
                                                                 num_total_days_per_patient,
                                                                 min_req_bs_sz_count,
                                                                 num_trials,
                                                                 placebo_mu,
                                                                 placebo_sigma,
                                                                 drug_mu,
                                                                 drug_sigma)
        
        '''        
        # did not change the code here to reflect that the mode is being calculated instead of the median
        median_percent_change_per_placebo_map_point = stats.mode(one_placebo_map_point_percent_changes)[0]
        median_percent_change_per_drug_map_point    = stats.mode(one_drug_map_point_percent_changes)[0]
        '''
        '''
        median_percent_change_per_placebo_map_point = np.percentile(one_placebo_map_point_percent_changes, 85)
        median_percent_change_per_drug_map_point    = np.percentile(one_drug_map_point_percent_changes, 85)
        '''
        
        median_percent_change_per_placebo_map_point = np.median(one_placebo_map_point_percent_changes)
        median_percent_change_per_drug_map_point    = np.median(one_drug_map_point_percent_changes)
        
        '''
        median_percent_change_per_placebo_map_point = np.mean(one_placebo_map_point_percent_changes)
        median_percent_change_per_drug_map_point    = np.mean(one_drug_map_point_percent_changes)
        '''

        average_TTP_per_placebo_map_point           = np.mean(one_placebo_map_point_TTP_times)
        average_TTP_per_drug_map_point              = np.mean(one_drug_map_point_TTP_times)

        median_percent_change_per_placebo_arm_theo_patients[theo_patient_index] = median_percent_change_per_placebo_map_point
        median_percent_change_per_drug_arm_theo_patients[theo_patient_index]    = median_percent_change_per_drug_map_point
        average_TTP_per_placebo_arm_theo_patients[theo_patient_index]           = average_TTP_per_placebo_map_point
        average_TTP_per_drug_arm_theo_patients[theo_patient_index]              = average_TTP_per_drug_map_point
        
        patient_point_stop_time_in_seconds = time.time()
        total_patient_point_runtime_in_seconds = patient_point_stop_time_in_seconds - patient_point_start_time_in_seconds
        total_patient_point_runtime_in_seconds_str = 'patient point #' + str(theo_patient_index + 1) + ' runtime: ' + str(np.round(total_patient_point_runtime_in_seconds, 3)) + ' seconds'
        print(total_patient_point_runtime_in_seconds_str)

    expected_ana_placebo_arm_RR50 = 100*np.sum(median_percent_change_per_placebo_arm_theo_patients >= 0.5)/num_theo_patients_per_trial_arm
    expected_ana_drug_arm_RR50    = 100*np.sum(median_percent_change_per_drug_arm_theo_patients >= 0.5)/num_theo_patients_per_trial_arm
    command = ['Rscript', 'Fisher_Exact_Power_Calc.R', str(expected_ana_placebo_arm_RR50), str(expected_ana_drug_arm_RR50), str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    fisher_exact_ana_stat_power = float(process.communicate()[0].decode().split()[1])

    return [expected_ana_placebo_arm_RR50, expected_ana_drug_arm_RR50, fisher_exact_ana_stat_power]


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_theo_patients_per_trial_arm = 153
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    min_req_bs_sz_count             = 4

    num_trials = 500

    num_baseline_days_per_patient = 28*num_baseline_months_per_patient
    num_testing_days_per_patient  = 28*num_testing_months_per_patient
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient 

    [patient_placebo_pop_monthly_params, patient_placebo_pop_daily_params] = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)
    
    [patient_drug_pop_monthly_params, patient_drug_pop_daily_params] = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)

    [expected_emp_placebo_arm_RR50, expected_emp_drug_arm_RR50, RR50_emp_stat_power, 
    fisher_exact_emp_stat_power,   MPC_emp_stat_power,          TTP_emp_stat_power] = \
        calculate_empirical_stat_power(patient_drug_pop_daily_params,
                                       patient_placebo_pop_daily_params, 
                                       num_theo_patients_per_trial_arm,
                                       num_baseline_days_per_patient,
                                       num_testing_days_per_patient,
                                       num_total_days_per_patient,
                                       min_req_bs_sz_count,
                                       placebo_mu,
                                       placebo_sigma,
                                       drug_mu,
                                       drug_sigma,
                                       num_trials)
    
    [expected_ana_placebo_arm_RR50, expected_ana_drug_arm_RR50, fisher_exact_ana_stat_power] = \
        calculate_analytical_stat_power(patient_placebo_pop_daily_params,
                                        patient_drug_pop_daily_params,
                                        num_theo_patients_per_trial_arm,
                                        num_baseline_days_per_patient,
                                        num_testing_days_per_patient,
                                        num_total_days_per_patient,
                                        min_req_bs_sz_count,
                                        num_trials,
                                        placebo_mu,
                                        placebo_sigma,
                                        drug_mu,
                                        drug_sigma)       

    expected_emp_placebo_arm_RR50_str = 'expected empirical placebo arm 50% responder rate :   ' + str(np.round(expected_emp_placebo_arm_RR50, 3)) + ' %'
    expected_emp_drug_arm_RR50_str    = 'expected empirical drug arm 50% responder rate :      ' + str(np.round(expected_emp_drug_arm_RR50, 3))    + ' %'
    expected_ana_placebo_arm_RR50_str = 'expected analytical placebo arm 50% responder rate :  ' + str(np.round(expected_ana_placebo_arm_RR50, 3)) + ' %'
    expected_ana_drug_arm_RR50_str    = 'expected analytical drug arm 50% responder rate :     ' + str(np.round(expected_ana_drug_arm_RR50, 3))    + ' %'

    RR50_emp_stat_power_str         = '50% responder rate empirical statistical power:       ' + str(np.round(RR50_emp_stat_power, 3))         + ' %'
    fisher_exact_emp_stat_power_str = 'Fisher Exact Test empirical statistical power:        ' + str(np.round(fisher_exact_emp_stat_power, 3)) + ' %'
    fisher_exact_ana_stat_power_str = 'Fisher Exact Test analytical statistical power:       ' + str(np.round(fisher_exact_ana_stat_power, 3)) + ' %'
    MPC_emp_stat_power_str          = 'median percent change empirical statistical power:    ' + str(np.round(MPC_emp_stat_power, 3))          + ' %'
    TTP_emp_stat_power_str          = 'time-to-prerandomization empirical statistical power: ' + str(np.round(TTP_emp_stat_power, 3))          + ' %'
    
    data_str = '\n' + expected_emp_placebo_arm_RR50_str + '\n' + expected_emp_drug_arm_RR50_str  + \
               '\n' + expected_ana_placebo_arm_RR50_str + '\n' + expected_ana_drug_arm_RR50_str  + '\n' + \
               '\n' + RR50_emp_stat_power_str           + '\n' + fisher_exact_ana_stat_power_str + \
               '\n' + MPC_emp_stat_power_str            + '\n' + TTP_emp_stat_power_str          + '\n'
    
    print(data_str)

