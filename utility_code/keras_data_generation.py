import numpy as np
import time
from patient_population_generation import generate_theo_patient_pop_params
from patient_population_generation import generate_heterogenous_placebo_arm_patient_pop
from patient_population_generation import generate_heterogenous_drug_arm_patient_pop
from patient_population_generation import convert_theo_pop_hist
from endpoint_functions import calculate_percent_changes
from endpoint_functions import calculate_time_to_prerandomizations
from endpoint_functions import calculate_fisher_exact_p_value
from endpoint_functions import calculate_Mann_Whitney_U_p_value
from endpoint_functions import calculate_logrank_p_value


def generate_theo_patient_pop_params_per_trial_arm(monthly_mean_min,
                                                   monthly_mean_max,
                                                   monthly_std_dev_min,
                                                   monthly_std_dev_max,
                                                   num_theo_patients_per_trial_arm):

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
    
    return [theo_placebo_arm_patient_pop_params, theo_drug_arm_patient_pop_params]


def generate_seizure_diaries_per_trial_arm(num_theo_patients_per_trial_arm,
                                           theo_placebo_arm_patient_pop_params,
                                           num_baseline_months,
                                           num_testing_months,
                                           baseline_time_scaling_const,
                                           testing_time_scaling_const,
                                           minimum_required_baseline_seizure_count,
                                           placebo_mu,
                                           placebo_sigma,
                                           drug_mu,
                                           drug_sigma):

    [placebo_arm_baseline_monthly_seizure_diaries, 
     placebo_arm_testing_daily_seizure_diaries  ] = \
         generate_heterogenous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                       theo_placebo_arm_patient_pop_params,
                                                       num_baseline_months,
                                                       num_testing_months,
                                                       baseline_time_scaling_const,
                                                       testing_time_scaling_const,
                                                       minimum_required_baseline_seizure_count,
                                                       placebo_mu,
                                                       placebo_sigma)

    [drug_arm_baseline_monthly_seizure_diaries, 
     drug_arm_testing_daily_seizure_diaries  ] = \
         generate_heterogenous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                    theo_placebo_arm_patient_pop_params,
                                                    num_baseline_months,
                                                    num_testing_months,
                                                    baseline_time_scaling_const,
                                                    testing_time_scaling_const,
                                                    minimum_required_baseline_seizure_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma)
    
    placebo_arm_testing_monthly_seizure_diaries = \
        np.sum(placebo_arm_testing_daily_seizure_diaries.reshape((num_theo_patients_per_trial_arm,
                                                                  num_testing_months,
                                                                  testing_time_scaling_const)), 2)
    
    drug_arm_testing_monthly_seizure_diaries = \
        np.sum(drug_arm_testing_daily_seizure_diaries.reshape((num_theo_patients_per_trial_arm,
                                                               num_testing_months, 
                                                               testing_time_scaling_const)), 2)

    return [placebo_arm_baseline_monthly_seizure_diaries,
            placebo_arm_testing_daily_seizure_diaries,
            placebo_arm_testing_monthly_seizure_diaries,
            drug_arm_baseline_monthly_seizure_diaries,
            drug_arm_testing_daily_seizure_diaries,
            drug_arm_testing_monthly_seizure_diaries]


def calculate_endpoints(num_testing_months,
                        testing_time_scaling_const,
                        num_theo_patients_per_trial_arm,
                        placebo_arm_baseline_monthly_seizure_diaries,
                        placebo_arm_testing_monthly_seizure_diaries,
                        placebo_arm_testing_daily_seizure_diaries,
                        drug_arm_baseline_monthly_seizure_diaries,
                        drug_arm_testing_monthly_seizure_diaries,
                        drug_arm_testing_daily_seizure_diaries):

    num_testing_days = num_testing_months*testing_time_scaling_const

    placebo_arm_percent_changes = \
            calculate_percent_changes(placebo_arm_baseline_monthly_seizure_diaries,
                                      placebo_arm_testing_monthly_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
    drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_monthly_seizure_diaries,
                                      drug_arm_testing_monthly_seizure_diaries,
                                      num_theo_patients_per_trial_arm)

    [placebo_arm_TTP_times, placebo_arm_observed_array] = \
            calculate_time_to_prerandomizations(placebo_arm_baseline_monthly_seizure_diaries,
                                                placebo_arm_testing_daily_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
    [drug_arm_TTP_times, drug_arm_observed_array] = \
            calculate_time_to_prerandomizations(drug_arm_baseline_monthly_seizure_diaries,
                                                drug_arm_testing_daily_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
    '''
    RR50_p_value = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)

    MPC_p_value = \
            calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                             drug_arm_percent_changes)
    
    TTP_p_value = \
            calculate_logrank_p_value(placebo_arm_TTP_times, 
                                      placebo_arm_observed_array, 
                                      drug_arm_TTP_times, 
                                      drug_arm_observed_array)
    '''
    
    return [placebo_arm_percent_changes, drug_arm_percent_changes,
            placebo_arm_TTP_times,       placebo_arm_observed_array,
            drug_arm_TTP_times,          drug_arm_observed_array]


def calculate_trial_successes(patient_num,
                              placebo_arm_percent_changes,
                              drug_arm_percent_changes,
                              placebo_arm_TTP_times,
                              placebo_arm_observed_array,
                              drug_arm_TTP_times,
                              drug_arm_observed_array):

    RR50_p_value = \
        calculate_fisher_exact_p_value(placebo_arm_percent_changes[0:patient_num],
                                       drug_arm_percent_changes[0:patient_num])

    MPC_p_value = \
        calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes[0:patient_num],
                                         drug_arm_percent_changes[0:patient_num])
    
    TTP_p_value = \
        calculate_logrank_p_value(placebo_arm_TTP_times[0:patient_num], 
                                  placebo_arm_observed_array[0:patient_num], 
                                  drug_arm_TTP_times[0:patient_num], 
                                  drug_arm_observed_array[0:patient_num])
    
    return [RR50_p_value, MPC_p_value, TTP_p_value]


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    max_theo_patients_per_trial_arm  = 300
    theo_patients_per_trial_arm_step = 50

    num_baseline_months = 2
    num_testing_months  = 3
    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 28
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials = 200

    algorithm_estim_start_time_in_seconds = time.time()

    [theo_placebo_arm_patient_pop_params, 
     theo_drug_arm_patient_pop_params] = \
         generate_theo_patient_pop_params_per_trial_arm(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        max_theo_patients_per_trial_arm)

    patient_nums = np.arange(theo_patients_per_trial_arm_step, max_theo_patients_per_trial_arm + theo_patients_per_trial_arm_step, theo_patients_per_trial_arm_step)
    num_trial_sizes = len(patient_nums)

    RR50_p_value_matrix = np.zeros((num_trial_sizes, num_trials))
    MPC_p_value_matrix  = np.zeros((num_trial_sizes, num_trials))
    TTP_p_value_matrix  = np.zeros((num_trial_sizes, num_trials))

    for trial_index in range(num_trials):

        endpoint_start_time_in_seconds = time.time()

        [placebo_arm_baseline_monthly_seizure_diaries,
         placebo_arm_testing_daily_seizure_diaries,
         placebo_arm_testing_monthly_seizure_diaries,
         drug_arm_baseline_monthly_seizure_diaries,
         drug_arm_testing_daily_seizure_diaries,
         drug_arm_testing_monthly_seizure_diaries] = \
             generate_seizure_diaries_per_trial_arm(max_theo_patients_per_trial_arm,
                                                    theo_placebo_arm_patient_pop_params,
                                                    num_baseline_months,
                                                    num_testing_months,
                                                    baseline_time_scaling_const,
                                                    testing_time_scaling_const,
                                                    minimum_required_baseline_seizure_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma)

        [placebo_arm_percent_changes, drug_arm_percent_changes,
         placebo_arm_TTP_times,       placebo_arm_observed_array,
         drug_arm_TTP_times,          drug_arm_observed_array] = \
             calculate_endpoints(num_testing_months,
                                 testing_time_scaling_const,
                                 max_theo_patients_per_trial_arm,
                                 placebo_arm_baseline_monthly_seizure_diaries,
                                 placebo_arm_testing_monthly_seizure_diaries,
                                 placebo_arm_testing_daily_seizure_diaries,
                                 drug_arm_baseline_monthly_seizure_diaries,
                                 drug_arm_testing_monthly_seizure_diaries,
                                 drug_arm_testing_daily_seizure_diaries)
        
        endpoint_stop_time_in_seconds = time.time()
        endpoint_calc_runtime_in_seconds_str = str(np.round(endpoint_stop_time_in_seconds - endpoint_start_time_in_seconds, 3))
        print( 'trial # ' + str(trial_index) + ' runtime: ' + endpoint_calc_runtime_in_seconds_str + ' seconds')
        
        for patient_num_index in range(num_trial_sizes):

            patient_num = patient_nums[patient_num_index]

            p_value_calc_start_time = time.time()

            [RR50_p_value, MPC_p_value, TTP_p_value] = \
                calculate_trial_successes(patient_num,
                                          placebo_arm_percent_changes[0:patient_num],
                                          drug_arm_percent_changes[0:patient_num],
                                          placebo_arm_TTP_times[0:patient_num],
                                          placebo_arm_observed_array[0:patient_num],
                                          drug_arm_TTP_times[0:patient_num],
                                          drug_arm_observed_array[0:patient_num])
            
            p_value_calc_stop_time = time.time()
            p_value_calc_runtime_in_seconds_str = str(np.round(p_value_calc_stop_time - p_value_calc_start_time, 3))
            print( 'trial size ' + str(patient_num) + ' runtime: ' + p_value_calc_runtime_in_seconds_str + ' seconds')

            RR50_p_value_matrix[patient_num_index, trial_index] = RR50_p_value < 0.05
            MPC_p_value_matrix[patient_num_index, trial_index]  = MPC_p_value  < 0.05
            TTP_p_value_matrix[patient_num_index, trial_index]  = TTP_p_value  < 0.05
    
    RR50_stat_powers = np.mean(RR50_p_value_matrix, 1)
    MPC_stat_powers  = np.mean(MPC_p_value_matrix,  1)
    TTP_stat_powers  = np.mean(TTP_p_value_matrix,  1)

    algorithm_estim_stop_time_in_seconds = time.time()

    import pandas as pd

    for patient_num_index in range(num_trial_sizes):

        patient_num = patient_nums[patient_num_index]

        theo_placebo_arm_patient_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  theo_placebo_arm_patient_pop_params[0:patient_num])
        
        theo_drug_arm_patient_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  theo_drug_arm_patient_pop_params[0:patient_num])
        
        print('\n\ntrial size: '            + str(patient_num)                                           + ' patients\n'
               '\nRR50 statistical power: ' + str(np.round(100*RR50_stat_powers[patient_num_index], 3))  + ' %\n'   + \
               '\n MPC statistical power: ' + str(np.round(100*MPC_stat_powers[patient_num_index],  3))  + ' %\n'   + \
               '\n TTP statistical power: ' + str(np.round(100*TTP_stat_powers[patient_num_index],  3))  + ' %\n\n' + \
               pd.DataFrame(theo_placebo_arm_patient_pop_hist).to_string()             +   '\n\n' + \
               pd.DataFrame(theo_drug_arm_patient_pop_hist).to_string()                +   '\n')
    
    print('algorithm runtime: ' + str(np.round(algorithm_estim_stop_time_in_seconds - algorithm_estim_start_time_in_seconds, 3)) + ' seconds')

