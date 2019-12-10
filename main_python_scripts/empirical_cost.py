import numpy as np
import copy
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_baseline_seizure_diary
from utility_code.patient_population_generation import generate_heterogeneous_placebo_arm_patient_pop
from utility_code.patient_population_generation import generate_heterogeneous_drug_arm_patient_pop
from utility_code.endpoint_functions import calculate_percent_changes
from utility_code.endpoint_functions import calculate_time_to_prerandomizations
from utility_code.endpoint_functions import calculate_fisher_exact_p_value
from utility_code.endpoint_functions import calculate_Mann_Whitney_U_p_value
from utility_code.endpoint_functions import calculate_logrank_p_value


def get_map_loc(monthly_mean_min,
                monthly_mean_max,
                monthly_std_dev_min,
                monthly_std_dev_max):

    overdispersed = False
    non_zero_mean = False

    while((not overdispersed) or (not non_zero_mean)):

        overdispersed = False
        non_zero_mean = False
        
        monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max    + 1)
        monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max + 1)

        if(monthly_mean != 0):

                non_zero_mean = True

        if(monthly_std_dev > np.sqrt(monthly_mean)):

            overdispersed = True
    
    return [monthly_mean, monthly_std_dev]


def estimate_patient_loc(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_baseline_months,
                         minimum_required_baseline_seizure_count):

    baseline_time_scaling_const = 28
    estims_within_SNR_map = False

    while(estims_within_SNR_map == False):

        [monthly_mean, 
         monthly_std_dev] = \
             get_map_loc(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max)

        daily_baseline_seizure_diary = \
            np.int_(generate_baseline_seizure_diary(monthly_mean, 
                                                    monthly_std_dev,
                                                    num_baseline_months,
                                                    baseline_time_scaling_const,
                                                    minimum_required_baseline_seizure_count))

        monthly_mean_hat    =          np.int_(np.round(baseline_time_scaling_const*np.mean(daily_baseline_seizure_diary)))
        monthly_std_dev_hat =  np.int_(np.round(np.sqrt(baseline_time_scaling_const)*np.std(daily_baseline_seizure_diary)))

        monthly_mean_hat_within_SNR_map = (monthly_mean_hat >= monthly_mean_min) and (monthly_mean_hat <= monthly_mean_max)
        monthly_std_dev_hat_within_SNR_map = (monthly_std_dev_hat >= monthly_std_dev_min) and (monthly_std_dev_hat <= monthly_std_dev_max)
        overdispersed_hat = np.power(monthly_std_dev_hat, 2) > monthly_mean_hat

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and overdispersed_hat

    return [monthly_mean_hat, monthly_std_dev_hat]


def add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                            monthly_mean_max,
                                            monthly_std_dev_min,
                                            monthly_std_dev_max,
                                            num_baseline_months,
                                            minimum_required_baseline_seizure_count,
                                            trial_arm_theo_patient_pop_list):

    [monthly_mean_hat, 
     monthly_std_dev_hat] = \
         estimate_patient_loc(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              num_baseline_months,
                              minimum_required_baseline_seizure_count)
    
    trial_arm_theo_patient_pop_list.append([monthly_mean_hat, monthly_std_dev_hat])

    return trial_arm_theo_patient_pop_list


def generate_percent_change_data(num_theo_patients_in_placebo_arm,
                                 theo_placebo_arm_patient_pop_params,
                                 num_theo_patients_in_drug_arm,
                                 theo_drug_arm_patient_pop_params,
                                 num_baseline_months,
                                 num_testing_months,
                                 minimum_required_baseline_seizure_count,
                                 placebo_mu,
                                 placebo_sigma,
                                 drug_mu,
                                 drug_sigma):

    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 1

    [placebo_arm_monthly_baseline_seizure_diaries, 
     placebo_arm_monthly_testing_seizure_diaries  ] = \
         generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_in_placebo_arm,
                                                        theo_placebo_arm_patient_pop_params,
                                                        num_baseline_months,
                                                        num_testing_months,
                                                        baseline_time_scaling_const,
                                                        testing_time_scaling_const,
                                                        minimum_required_baseline_seizure_count,
                                                        placebo_mu,
                                                        placebo_sigma)
    
    [drug_arm_monthly_baseline_seizure_diaries, 
     drug_arm_monthly_testing_seizure_diaries  ] = \
         generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_in_drug_arm,
                                                     theo_drug_arm_patient_pop_params,
                                                     num_baseline_months,
                                                     num_testing_months,
                                                     baseline_time_scaling_const,
                                                     testing_time_scaling_const,
                                                     minimum_required_baseline_seizure_count,
                                                     placebo_mu,
                                                     placebo_sigma,
                                                     drug_mu,
                                                     drug_sigma)
    
    placebo_arm_percent_changes = \
        calculate_percent_changes(placebo_arm_monthly_baseline_seizure_diaries,
                                  placebo_arm_monthly_testing_seizure_diaries)

    drug_arm_percent_changes = \
        calculate_percent_changes(drug_arm_monthly_baseline_seizure_diaries,
                                  drug_arm_monthly_testing_seizure_diaries)
    
    return [placebo_arm_percent_changes, 
            drug_arm_percent_changes]


def generate_prerandomization_data(num_theo_patients_in_placebo_arm,
                                   theo_placebo_arm_patient_pop_params,
                                   num_theo_patients_in_drug_arm,
                                   theo_drug_arm_patient_pop_params,
                                   num_baseline_months,
                                   num_testing_months,
                                   minimum_required_baseline_seizure_count,
                                   placebo_mu,
                                   placebo_sigma,
                                   drug_mu,
                                   drug_sigma):

    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 28
    num_testing_days = testing_time_scaling_const*num_testing_months

    [placebo_arm_monthly_baseline_seizure_diaries, 
     placebo_arm_daily_testing_seizure_diaries  ] = \
         generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_in_placebo_arm,
                                                        theo_placebo_arm_patient_pop_params,
                                                        num_baseline_months,
                                                        num_testing_months,
                                                        baseline_time_scaling_const,
                                                        testing_time_scaling_const,
                                                        minimum_required_baseline_seizure_count,
                                                        placebo_mu,
                                                        placebo_sigma)
    
    [drug_arm_monthly_baseline_seizure_diaries, 
     drug_arm_daily_testing_seizure_diaries  ] = \
         generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_in_drug_arm,
                                                     theo_drug_arm_patient_pop_params,
                                                     num_baseline_months,
                                                     num_testing_months,
                                                     baseline_time_scaling_const,
                                                     testing_time_scaling_const,
                                                     minimum_required_baseline_seizure_count,
                                                     placebo_mu,
                                                     placebo_sigma,
                                                     drug_mu,
                                                     drug_sigma)

    [placebo_arm_TTP_times, 
     placebo_arm_observed_array] = \
         calculate_time_to_prerandomizations(placebo_arm_monthly_baseline_seizure_diaries,
                                             placebo_arm_daily_testing_seizure_diaries,
                                             num_theo_patients_in_placebo_arm,
                                             num_testing_days)

    [drug_arm_TTP_times, 
     drug_arm_observed_array] = \
         calculate_time_to_prerandomizations(drug_arm_monthly_baseline_seizure_diaries,
                                             drug_arm_daily_testing_seizure_diaries,
                                             num_theo_patients_in_drug_arm,
                                             num_testing_days)

    return [placebo_arm_TTP_times, 
            placebo_arm_observed_array,
            drug_arm_TTP_times, 
            drug_arm_observed_array]              


def generate_trial_p_value(placebo_arm_theo_patient_pop_list,
                           drug_arm_theo_patient_pop_list,
                           endpoint_name,
                           num_baseline_months,
                           num_testing_months,
                           minimum_required_baseline_seizure_count,
                           placebo_mu,
                           placebo_sigma,
                           drug_mu,
                           drug_sigma):

    num_theo_patients_in_placebo_arm = len(placebo_arm_theo_patient_pop_list)
    num_theo_patients_in_drug_arm    = len(drug_arm_theo_patient_pop_list)

    theo_placebo_arm_patient_pop_params = np.array(placebo_arm_theo_patient_pop_list)
    theo_drug_arm_patient_pop_params    = np.array(drug_arm_theo_patient_pop_list)

    if(endpoint_name == 'TTP'):
    
        [placebo_arm_TTP_times, 
         placebo_arm_observed_array,
         drug_arm_TTP_times, 
         drug_arm_observed_array]    = \
             generate_prerandomization_data(num_theo_patients_in_placebo_arm,
                                            theo_placebo_arm_patient_pop_params,
                                            num_theo_patients_in_drug_arm,
                                            theo_drug_arm_patient_pop_params,
                                            num_baseline_months,
                                            num_testing_months,
                                            minimum_required_baseline_seizure_count,
                                            placebo_mu,
                                            placebo_sigma,
                                            drug_mu,
                                            drug_sigma)
        
        TTP_p_value = \
            calculate_logrank_p_value(placebo_arm_TTP_times, 
                                      placebo_arm_observed_array, 
                                      drug_arm_TTP_times, 
                                      drug_arm_observed_array)
        
        return TTP_p_value

    else:

        [placebo_arm_percent_changes, 
         drug_arm_percent_changes] = \
             generate_percent_change_data(num_theo_patients_in_placebo_arm,
                                          theo_placebo_arm_patient_pop_params,
                                          num_theo_patients_in_drug_arm,
                                          theo_drug_arm_patient_pop_params,
                                          num_baseline_months,
                                          num_testing_months,
                                          minimum_required_baseline_seizure_count,
                                          placebo_mu,
                                          placebo_sigma,
                                          drug_mu,
                                          drug_sigma)
        
        if(endpoint_name == 'RR50'):

            RR50_p_value = \
                calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                               drug_arm_percent_changes)
        
            return RR50_p_value

        elif(endpoint_name == 'MPC'):

            MPC_p_value = \
                calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                                 drug_arm_percent_changes)
            
            return MPC_p_value


def estimate_statistical_power(placebo_arm_theo_patient_pop_list,
                               drug_arm_theo_patient_pop_list,
                               endpoint_name,
                               num_baseline_months,
                               num_testing_months,
                               minimum_required_baseline_seizure_count,
                               placebo_mu,
                               placebo_sigma,
                               drug_mu,
                               drug_sigma,
                               num_trials):

    p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        p_value = \
            generate_trial_p_value(placebo_arm_theo_patient_pop_list,
                                   drug_arm_theo_patient_pop_list,
                                   endpoint_name,
                                   num_baseline_months,
                                   num_testing_months,
                                   minimum_required_baseline_seizure_count,
                                   placebo_mu,
                                   placebo_sigma,
                                   drug_mu,
                                   drug_sigma)

        p_value_array[trial_index] = p_value

    stat_power = np.sum(p_value_array <= 0.05)/num_trials

    return stat_power


def dumb_algorithm(monthly_mean_min,
                   monthly_mean_max,
                   monthly_std_dev_min,
                   monthly_std_dev_max,endpoint_name,
                   num_baseline_months,
                   num_testing_months,
                   minimum_required_baseline_seizure_count,
                   placebo_mu,
                   placebo_sigma,
                   drug_mu,
                   drug_sigma,
                   target_stat_power,
                   num_trials):

    stat_power = 0
    trial_arm  = 'placebo'

    placebo_arm_theo_patient_pop_list = []
    drug_arm_theo_patient_pop_list    = []

    placebo_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                num_baseline_months,
                                                minimum_required_baseline_seizure_count,
                                                placebo_arm_theo_patient_pop_list)

    drug_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                num_baseline_months,
                                                minimum_required_baseline_seizure_count,
                                                drug_arm_theo_patient_pop_list)
    
    while(stat_power < target_stat_power):

        if(trial_arm == 'placebo'):

            placebo_arm_theo_patient_pop_list = \
                add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        num_baseline_months,
                                                        minimum_required_baseline_seizure_count,
                                                        placebo_arm_theo_patient_pop_list)
            
            trial_arm = 'drug'

        elif(trial_arm == 'drug'):

            drug_arm_theo_patient_pop_list = \
                add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        num_baseline_months,
                                                        minimum_required_baseline_seizure_count,
                                                        drug_arm_theo_patient_pop_list)
            
            trial_arm = 'placebo'

        stat_power = \
            estimate_statistical_power(placebo_arm_theo_patient_pop_list,
                                       drug_arm_theo_patient_pop_list,
                                       endpoint_name,
                                       num_baseline_months,
                                       num_testing_months,
                                       minimum_required_baseline_seizure_count,
                                       placebo_mu,
                                       placebo_sigma,
                                       drug_mu,
                                       drug_sigma,
                                       num_trials)

        print('\nstatistical power: ' + str(np.round(100*stat_power, 3)) + ' %' + \
              ', total number of patients: ' + str(len(placebo_arm_theo_patient_pop_list) + len(drug_arm_theo_patient_pop_list)) + \
              '\nnumber of placebo arm patients: ' + str(len(placebo_arm_theo_patient_pop_list)) + \
              '\nnumber of drug arm patients: ' + str(len(drug_arm_theo_patient_pop_list)))


if(__name__=='__main__'):

    # SNR map parameters
    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    # RCT design parameters
    endpoint_name = sys.argv[1]
    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4

    # simulation parameters
    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05
    target_stat_power = 0.9

    # computational estimation parameters
    num_trials = 2000

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#

    stat_power = 0
    trial_arm  = 'placebo'

    placebo_arm_theo_patient_pop_list = []
    drug_arm_theo_patient_pop_list    = []

    placebo_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                num_baseline_months,
                                                minimum_required_baseline_seizure_count,
                                                placebo_arm_theo_patient_pop_list)

    drug_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                num_baseline_months,
                                                minimum_required_baseline_seizure_count,
                                                drug_arm_theo_patient_pop_list)
    
    while(stat_power < target_stat_power):

        tmp_placebo_arm_theo_patient_pop_list = \
            copy.deepcopy(placebo_arm_theo_patient_pop_list)
        tmp_drug_arm_theo_patient_pop_list = \
            copy.deepcopy(drug_arm_theo_patient_pop_list)

        if(trial_arm == 'placebo'):

            tmp_placebo_arm_theo_patient_pop_list = \
                add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        num_baseline_months,
                                                        minimum_required_baseline_seizure_count,
                                                        tmp_placebo_arm_theo_patient_pop_list)
            
            trial_arm = 'drug'

        elif(trial_arm == 'drug'):

            tmp_drug_arm_theo_patient_pop_list = \
                add_patient_to_placebo_theo_patient_pop(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        num_baseline_months,
                                                        minimum_required_baseline_seizure_count,
                                                        tmp_drug_arm_theo_patient_pop_list)
            
            trial_arm = 'placebo'

        tmp_stat_power = \
            estimate_statistical_power(tmp_placebo_arm_theo_patient_pop_list,
                                       tmp_drug_arm_theo_patient_pop_list,
                                       endpoint_name,
                                       num_baseline_months,
                                       num_testing_months,
                                       minimum_required_baseline_seizure_count,
                                       placebo_mu,
                                       placebo_sigma,
                                       drug_mu,
                                       drug_sigma,
                                       num_trials)

        SNR = tmp_stat_power - stat_power

        if(SNR >= 0):

            placebo_arm_theo_patient_pop_list = tmp_placebo_arm_theo_patient_pop_list
            drug_arm_theo_patient_pop_list    = tmp_drug_arm_theo_patient_pop_list
            stat_power = tmp_stat_power

        print('\nSNR: ' + str(np.round(100*SNR, 3)) + ', trial_arm: ' + trial_arm + \
              '\nstatistical power: ' + str(np.round(100*stat_power, 3)) + ' %' + \
              ', total number of patients: ' + str(len(placebo_arm_theo_patient_pop_list) + len(drug_arm_theo_patient_pop_list)) + \
              '\nnumber of placebo arm patients: ' + str(len(placebo_arm_theo_patient_pop_list)) + \
              '\nnumber of drug arm patients: ' + str(len(drug_arm_theo_patient_pop_list)) + '\n')

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#

