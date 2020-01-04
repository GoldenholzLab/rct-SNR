import numpy as np
import time
import json
import os
import sys
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import randomly_select_theo_patient_pop
from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import generate_heterogeneous_placebo_arm_patient_pop
from utility_code.patient_population_generation import generate_heterogeneous_drug_arm_patient_pop
from utility_code.patient_population_generation import convert_theo_pop_hist
from utility_code.endpoint_functions import calculate_percent_changes
from utility_code.endpoint_functions import calculate_time_to_prerandomizations
from utility_code.endpoint_functions import calculate_fisher_exact_p_value
from utility_code.endpoint_functions import calculate_Mann_Whitney_U_p_value
from utility_code.endpoint_functions import calculate_logrank_p_value



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
                                           theo_drug_arm_patient_pop_params,
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
         generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
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
         generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
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
                                      placebo_arm_testing_monthly_seizure_diaries)
    
    drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_monthly_seizure_diaries,
                                      drug_arm_testing_monthly_seizure_diaries)

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


def generate_powers_and_histograms(monthly_mean_lower_bound,
                                   monthly_mean_upper_bound,
                                   monthly_std_dev_lower_bound,
                                   monthly_std_dev_upper_bound,
                                   max_theo_patients_per_trial_arm,
                                   theo_patients_per_trial_arm_step,
                                   num_baseline_months,
                                   num_testing_months,
                                   baseline_time_scaling_const,
                                   testing_time_scaling_const,
                                   minimum_required_baseline_seizure_count,
                                   placebo_mu,
                                   placebo_sigma,
                                   drug_mu,
                                   drug_sigma,
                                   num_trials):

    algorithm_start_time_in_seconds = time.time()

    [monthly_mean_min, 
     monthly_mean_max,
     monthly_std_dev_min, 
     monthly_std_dev_max] = \
         randomly_select_theo_patient_pop(monthly_mean_lower_bound,
                                          monthly_mean_upper_bound,
                                          monthly_std_dev_lower_bound,
                                          monthly_std_dev_upper_bound)

    print('\n' + str([monthly_mean_min, monthly_mean_max, monthly_std_dev_min, monthly_std_dev_max]) + '\n')

    [theo_placebo_arm_patient_pop_params, 
     theo_drug_arm_patient_pop_params] = \
         generate_theo_patient_pop_params_per_trial_arm(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        max_theo_patients_per_trial_arm)

    patient_nums = np.arange(theo_patients_per_trial_arm_step, max_theo_patients_per_trial_arm + theo_patients_per_trial_arm_step, theo_patients_per_trial_arm_step)
    num_trial_arm_sizes = len(patient_nums)

    RR50_p_value_matrix = np.zeros((num_trial_arm_sizes, num_trials))
    MPC_p_value_matrix  = np.zeros((num_trial_arm_sizes, num_trials))
    TTP_p_value_matrix  = np.zeros((num_trial_arm_sizes, num_trials))

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
        print( 'trial # ' + str(trial_index + 1) + ' runtime: ' + endpoint_calc_runtime_in_seconds_str + ' seconds')
        
        for patient_num_index in range(num_trial_arm_sizes):

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
            print( 'trial arm size ' + str(patient_num) + ' runtime: ' + p_value_calc_runtime_in_seconds_str + ' seconds')

            RR50_p_value_matrix[patient_num_index, trial_index] = RR50_p_value < 0.05
            MPC_p_value_matrix[patient_num_index, trial_index]  = MPC_p_value  < 0.05
            TTP_p_value_matrix[patient_num_index, trial_index]  = TTP_p_value  < 0.05

        algorithm_stop_time_in_seconds = time.time()
        algorithm_cumulative_runtime_in_minutes_str = str(np.round((algorithm_stop_time_in_seconds - algorithm_start_time_in_seconds)/60, 3))
        print('algorithm cumulative runtime: ' + algorithm_cumulative_runtime_in_minutes_str + ' minutes')
    
    RR50_stat_powers = np.mean(RR50_p_value_matrix, 1)
    MPC_stat_powers  = np.mean(MPC_p_value_matrix,  1)
    TTP_stat_powers  = np.mean(TTP_p_value_matrix,  1)

    num_monthly_std_devs = monthly_std_dev_upper_bound - monthly_std_dev_lower_bound + 1
    num_monthly_means    = monthly_mean_upper_bound    - monthly_mean_lower_bound    + 1
    theo_placebo_arm_patient_pop_hists = np.zeros((num_monthly_std_devs, num_monthly_means, num_trial_arm_sizes))
    theo_drug_arm_patient_pop_hists    = np.zeros((num_monthly_std_devs, num_monthly_means, num_trial_arm_sizes))

    for patient_num_index in range(num_trial_arm_sizes):

        patient_num = patient_nums[patient_num_index]

        theo_placebo_arm_patient_pop_hists[:, :, patient_num_index] = \
            convert_theo_pop_hist(monthly_mean_lower_bound,
                                  monthly_mean_upper_bound,
                                  monthly_std_dev_lower_bound,
                                  monthly_std_dev_upper_bound,
                                  theo_placebo_arm_patient_pop_params[0:patient_num])
        
        theo_drug_arm_patient_pop_hists[:, :, patient_num_index] = \
            convert_theo_pop_hist(monthly_mean_lower_bound,
                                  monthly_mean_upper_bound,
                                  monthly_std_dev_lower_bound,
                                  monthly_std_dev_upper_bound,
                                  theo_drug_arm_patient_pop_params[0:patient_num])
    
    algorithm_stop_time_in_seconds = time.time()
    algorithm_runtime_in_minutes_str = str(np.round((algorithm_stop_time_in_seconds - algorithm_start_time_in_seconds)/60, 3))
    print('algorithm runtime: ' + algorithm_runtime_in_minutes_str + ' minutes')

    return [RR50_stat_powers, MPC_stat_powers, TTP_stat_powers, 
            theo_placebo_arm_patient_pop_hists, 
            theo_drug_arm_patient_pop_hists]


def store_powers_and_histograms(data_storage_folder_name,
                                file_index_str,
                                block_num_str,
                                RR50_stat_powers,
                                MPC_stat_powers,
                                TTP_stat_powers,
                                theo_placebo_arm_patient_pop_hists,
                                theo_drug_arm_patient_pop_hists):

    folder = data_storage_folder_name + '_' + block_num_str

    if( not os.path.isdir(folder) ):

        os.mkdir(folder)

    with open(folder + '/RR50_emp_stat_powers_' + file_index_str + '.json', 'w+') as json_file:

        json.dump(RR50_stat_powers.tolist(), json_file)
    
    with open(folder + '/MPC_emp_stat_powers_' + file_index_str + '.json', 'w+') as json_file:

        json.dump(MPC_stat_powers.tolist(), json_file)
    
    with open(folder + '/TTP_emp_stat_powers_' + file_index_str + '.json', 'w+') as json_file:

        json.dump(TTP_stat_powers.tolist(), json_file)
    
    with open(folder + '/theo_placebo_arm_hists_' + file_index_str + '.json', 'w+') as json_file:

        json.dump(theo_placebo_arm_patient_pop_hists.tolist(), json_file)
    
    with open(folder + '/theo_drug_arm_hists_' + file_index_str + '.json', 'w+') as json_file:

        json.dump(theo_drug_arm_patient_pop_hists.tolist(), json_file)


def take_input_arguments_from_command_shell():

    monthly_mean_lower_bound    = int(sys.argv[1])
    monthly_mean_upper_bound    = int(sys.argv[2])
    monthly_std_dev_lower_bound = int(sys.argv[3])
    monthly_std_dev_upper_bound = int(sys.argv[4])

    max_theo_patients_per_trial_arm  = int(sys.argv[5])
    theo_patients_per_trial_arm_step = int(sys.argv[6])

    num_baseline_months = int(sys.argv[7])
    num_testing_months  = int(sys.argv[8])
    baseline_time_scaling_const = int(sys.argv[9])
    testing_time_scaling_const  = int(sys.argv[10])
    minimum_required_baseline_seizure_count = int(sys.argv[11])

    placebo_mu    = float(sys.argv[12])
    placebo_sigma = float(sys.argv[13])
    drug_mu       = float(sys.argv[14])
    drug_sigma    = float(sys.argv[15])

    num_trials = int(sys.argv[16])

    block_num_str  = sys.argv[17]
    data_storage_folder_name = sys.argv[18]
    file_index_str = sys.argv[19]

    return [monthly_mean_lower_bound,    
            monthly_mean_upper_bound,
            monthly_std_dev_lower_bound, 
            monthly_std_dev_upper_bound,
            max_theo_patients_per_trial_arm,
            theo_patients_per_trial_arm_step,
            num_baseline_months, num_testing_months,
            baseline_time_scaling_const,
            testing_time_scaling_const,
            minimum_required_baseline_seizure_count,
            placebo_mu, placebo_sigma,
            drug_mu,    drug_sigma,
            num_trials, block_num_str, 
            data_storage_folder_name, file_index_str]


if(__name__=='__main__'):

    [monthly_mean_lower_bound,    
     monthly_mean_upper_bound,
     monthly_std_dev_lower_bound, 
     monthly_std_dev_upper_bound,
     max_theo_patients_per_trial_arm,
     theo_patients_per_trial_arm_step,
     num_baseline_months, num_testing_months,
     baseline_time_scaling_const,
     testing_time_scaling_const,
     minimum_required_baseline_seizure_count,
     placebo_mu, placebo_sigma,
     drug_mu,    drug_sigma,
     num_trials, block_num_str, 
     data_storage_folder_name, file_index_str] = \
         take_input_arguments_from_command_shell()

    print('\n' + data_storage_folder_name + '\n\nblock #' + block_num_str + '\n')

    [RR50_stat_powers, MPC_stat_powers, TTP_stat_powers, 
     theo_placebo_arm_patient_pop_hists, 
     theo_drug_arm_patient_pop_hists] = \
         generate_powers_and_histograms(monthly_mean_lower_bound,
                                        monthly_mean_upper_bound,
                                        monthly_std_dev_lower_bound,
                                        monthly_std_dev_upper_bound,
                                        max_theo_patients_per_trial_arm,
                                        theo_patients_per_trial_arm_step,
                                        num_baseline_months,
                                        num_testing_months,
                                        baseline_time_scaling_const,
                                        testing_time_scaling_const,
                                        minimum_required_baseline_seizure_count,
                                        placebo_mu,
                                        placebo_sigma,
                                        drug_mu,
                                        drug_sigma,
                                        num_trials)
    
    store_powers_and_histograms(data_storage_folder_name,
                                file_index_str,
                                block_num_str,
                                RR50_stat_powers,
                                MPC_stat_powers,
                                TTP_stat_powers,
                                theo_placebo_arm_patient_pop_hists,
                                theo_drug_arm_patient_pop_hists)

