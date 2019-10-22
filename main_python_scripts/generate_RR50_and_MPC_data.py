import numpy as np
import json
import time
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import randomly_select_theo_patient_pop
from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import convert_theo_pop_hist_with_empty_regions
from utility_code.empirical_estimation import empirically_estimate_RR50_and_MPC_statistical_power



def generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_lower_bound,
                                                  monthly_mean_upper_bound,
                                                  monthly_std_dev_lower_bound,
                                                  monthly_std_dev_upper_bound,
                                                  num_theo_patients_per_trial_arm):

    [monthly_mean_min, 
     monthly_mean_max, 
     monthly_std_dev_min, 
     monthly_std_dev_max] = \
         randomly_select_theo_patient_pop(monthly_mean_lower_bound,
                                          monthly_mean_upper_bound,
                                          monthly_std_dev_lower_bound,
                                          monthly_std_dev_upper_bound)

    theo_trial_arm_patient_pop_params = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    theo_trial_arm_pop_hist = \
        convert_theo_pop_hist_with_empty_regions(monthly_mean_lower_bound,
                                                  monthly_mean_upper_bound,
                                                  monthly_std_dev_lower_bound,
                                                  monthly_std_dev_upper_bound,
                                                  theo_trial_arm_patient_pop_params)
    
    return [theo_trial_arm_patient_pop_params, theo_trial_arm_pop_hist]


def estimate_statistical_powers_per_placebo_and_drug_theo_pops(theo_placebo_arm_patient_pop_params,
                                                               theo_drug_arm_patient_pop_params,
                                                               num_theo_patients_per_trial_arm,
                                                               num_baseline_months,
                                                               num_testing_months,
                                                               minimum_required_baseline_seizure_count,
                                                               placebo_mu,
                                                               placebo_sigma,
                                                               drug_mu,
                                                               drug_sigma,
                                                               num_trials):

    empirical_estimation_start_time_in_seconds = time.time()

    [RR50_emp_stat_power, MPC_emp_stat_power] = \
        empirically_estimate_RR50_and_MPC_statistical_power(theo_placebo_arm_patient_pop_params,
                                                            theo_drug_arm_patient_pop_params,
                                                            num_theo_patients_per_trial_arm,
                                                            num_baseline_months,
                                                            num_testing_months,
                                                            minimum_required_baseline_seizure_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            drug_mu,
                                                            drug_sigma,
                                                            num_trials)
    
    RR50_emp_stat_power_str = 'RR50 statistical power: ' + str(np.round(100*RR50_emp_stat_power, 3)) + ' %'
    MPC_emp_stat_power_str  = 'MPC  statistical power: ' + str(np.round(100*MPC_emp_stat_power,  3)) + ' %'
    
    empirical_estimation_stop_time_in_seconds = time.time()
    empirical_estimation_total_runtime_in_seconds = empirical_estimation_stop_time_in_seconds - empirical_estimation_start_time_in_seconds
    empirical_estimation_total_runtime_in_minutes = empirical_estimation_total_runtime_in_seconds/60
    empirical_estimation_total_runtime_in_minutes_str = 'empirical estimation runtime: ' + str(np.round(empirical_estimation_total_runtime_in_minutes , 3)) + ' minutes'
    print('\n' + empirical_estimation_total_runtime_in_minutes_str + '\n' + RR50_emp_stat_power_str + '\n' + MPC_emp_stat_power_str + '\n')

    return [RR50_emp_stat_power, MPC_emp_stat_power]


def generate_theo_pop_hists_and_powers_per_pop(monthly_mean_lower_bound,
                                               monthly_mean_upper_bound,
                                               monthly_std_dev_lower_bound,
                                               monthly_std_dev_upper_bound,
                                               num_theo_patients_per_trial_arm,
                                               num_baseline_months,
                                               num_testing_months,
                                               minimum_required_baseline_seizure_count,
                                               placebo_mu,
                                               placebo_sigma,
                                               drug_mu,
                                               drug_sigma,
                                               num_trials):

    [theo_placebo_arm_patient_pop_params, 
     theo_placebo_arm_pop_hist            ] = \
         generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_lower_bound,
                                                       monthly_mean_upper_bound,
                                                       monthly_std_dev_lower_bound,
                                                       monthly_std_dev_upper_bound,
                                                       num_theo_patients_per_trial_arm)
    
    [theo_drug_arm_patient_pop_params, 
     theo_drug_arm_pop_hist            ] = \
         generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_lower_bound,
                                                       monthly_mean_upper_bound,
                                                       monthly_std_dev_lower_bound,
                                                       monthly_std_dev_upper_bound,
                                                       num_theo_patients_per_trial_arm)

    [RR50_emp_stat_power, MPC_emp_stat_power] = \
        estimate_statistical_powers_per_placebo_and_drug_theo_pops(theo_placebo_arm_patient_pop_params,
                                                                   theo_drug_arm_patient_pop_params,
                                                                   num_theo_patients_per_trial_arm,
                                                                   num_baseline_months,
                                                                   num_testing_months,
                                                                   minimum_required_baseline_seizure_count,
                                                                   placebo_mu,
                                                                   placebo_sigma,
                                                                   drug_mu,
                                                                   drug_sigma,
                                                                   num_trials)
    
    return [theo_placebo_arm_pop_hist, theo_drug_arm_pop_hist, 
            RR50_emp_stat_power,       MPC_emp_stat_power     ]


def generate_theo_pop_hists_and_powers(monthly_mean_lower_bound,
                                       monthly_mean_upper_bound,
                                       monthly_std_dev_lower_bound,
                                       monthly_std_dev_upper_bound,
                                       num_theo_patients_per_trial_arm,
                                       num_baseline_months,
                                       num_testing_months,
                                       minimum_required_baseline_seizure_count,
                                       placebo_mu,
                                       placebo_sigma,
                                       drug_mu,
                                       drug_sigma,
                                       num_trials_per_pop,
                                       num_pops):
    
    num_monthly_means    = monthly_mean_upper_bound    - (monthly_mean_lower_bound    - 1)
    num_monthly_std_devs = monthly_std_dev_upper_bound - (monthly_std_dev_lower_bound - 1)

    theo_placebo_arm_hists = np.zeros((num_monthly_std_devs, num_monthly_means, num_pops))
    theo_drug_arm_hists    = np.zeros((num_monthly_std_devs, num_monthly_means, num_pops))
    RR50_emp_stat_powers   = np.zeros(num_pops)
    MPC_emp_stat_powers    = np.zeros(num_pops)
    
    for pop_index in range(num_pops):

        [theo_placebo_arm_pop_hist, theo_drug_arm_pop_hist, 
         RR50_emp_stat_power,       MPC_emp_stat_power     ] = \
             generate_theo_pop_hists_and_powers_per_pop(monthly_mean_lower_bound,
                                                        monthly_mean_upper_bound,
                                                        monthly_std_dev_lower_bound,
                                                        monthly_std_dev_upper_bound,
                                                        num_theo_patients_per_trial_arm,
                                                        num_baseline_months,
                                                        num_testing_months,
                                                        minimum_required_baseline_seizure_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma,
                                                        num_trials_per_pop)
    
        theo_placebo_arm_hists[:, :, pop_index] = theo_placebo_arm_pop_hist
        theo_drug_arm_hists[:, :, pop_index]    = theo_drug_arm_pop_hist
        RR50_emp_stat_powers[pop_index]         = RR50_emp_stat_power
        MPC_emp_stat_powers[pop_index]          = MPC_emp_stat_power
    
    return [theo_placebo_arm_hists, theo_drug_arm_hists, 
            RR50_emp_stat_powers,   MPC_emp_stat_powers]


def store_theo_pop_hists_and_emp_stat_powers(theo_placebo_arm_hists,
                                             theo_drug_arm_hists,
                                             RR50_emp_stat_powers,
                                             MPC_emp_stat_powers,
                                             data_storage_super_folder_path,
                                             block_generic_folder_name,
                                             compute_iter,
                                             block_num):
                                             
    data_storage_folder_file_path = data_storage_super_folder_path + '/' + block_generic_folder_name + '_' + str(int(block_num))

    if( not os.path.isdir(data_storage_folder_file_path) ):
        try:
            os.makedirs(data_storage_folder_file_path)
        except FileExistsError as err:
            print('\nran into timing hazard scenatio: \n' + str(err))

    theo_placebo_arm_hists_file_name = 'theo_placebo_arm_hists_' + str(compute_iter) + '.json'
    theo_drug_arm_hists_file_name    = 'theo_drug_arm_hists_'    + str(compute_iter) + '.json'
    RR50_emp_stat_powers_file_name   = 'RR50_emp_stat_powers_'   + str(compute_iter) + '.json'
    MPC_emp_stat_powers_file_name    = 'MPC_emp_stat_powers_'    + str(compute_iter) + '.json'
    
    theo_placebo_arm_hists_file_path = data_storage_folder_file_path + '/' + theo_placebo_arm_hists_file_name
    theo_drug_arm_hists_file_path    = data_storage_folder_file_path + '/' + theo_drug_arm_hists_file_name
    RR50_emp_stat_powers_file_path   = data_storage_folder_file_path + '/' + RR50_emp_stat_powers_file_name
    MPC_emp_stat_powers_file_path    = data_storage_folder_file_path + '/' + MPC_emp_stat_powers_file_name
    
    with open(theo_placebo_arm_hists_file_path, 'w+') as json_file:
        json.dump(theo_placebo_arm_hists.tolist(), json_file)
    
    with open(theo_drug_arm_hists_file_path, 'w+') as json_file:
        json.dump(theo_drug_arm_hists.tolist(), json_file)
    
    with open(RR50_emp_stat_powers_file_path, 'w+') as json_file:
        json.dump(RR50_emp_stat_powers.tolist(), json_file)
    
    with open(MPC_emp_stat_powers_file_path, 'w+') as json_file:
        json.dump(MPC_emp_stat_powers.tolist(), json_file)


def take_inputs_from_command_shell():

    monthly_mean_lower_bound    = int(sys.argv[1])
    monthly_mean_upper_bound    = int(sys.argv[2])
    monthly_std_dev_lower_bound = int(sys.argv[3])
    monthly_std_dev_upper_bound = int(sys.argv[4])

    num_theo_patients_per_trial_arm = int(sys.argv[5])
    num_baseline_months = int(sys.argv[6])
    num_testing_months  = int(sys.argv[7])
    minimum_required_baseline_seizure_count = int(sys.argv[8])

    placebo_mu    = float(sys.argv[9])
    placebo_sigma = float(sys.argv[10])
    drug_mu       = float(sys.argv[11])
    drug_sigma    = float(sys.argv[12])

    num_trials_per_pop = int(sys.argv[13])
    num_pops = int(sys.argv[14])

    data_storage_super_folder_path = sys.argv[15]
    block_generic_folder_name = sys.argv[16]

    block_num = int(sys.argv[17])
    compute_iter = int(sys.argv[18])


    return [monthly_mean_lower_bound, monthly_mean_upper_bound, 
            monthly_std_dev_lower_bound, monthly_std_dev_upper_bound,
            num_theo_patients_per_trial_arm, num_baseline_months, num_testing_months, 
            minimum_required_baseline_seizure_count,
            placebo_mu, placebo_sigma, drug_mu, drug_sigma,
            num_trials_per_pop, num_pops, 
            data_storage_super_folder_path, block_generic_folder_name, compute_iter, block_num]


if(__name__=='__main__'):

    [monthly_mean_lower_bound, monthly_mean_upper_bound, 
     monthly_std_dev_lower_bound, monthly_std_dev_upper_bound,
     num_theo_patients_per_trial_arm, num_baseline_months, num_testing_months, 
     minimum_required_baseline_seizure_count,
     placebo_mu, placebo_sigma, drug_mu, drug_sigma,
     num_trials_per_pop, num_pops, 
     data_storage_super_folder_path, block_generic_folder_name, compute_iter, block_num] = \
         take_inputs_from_command_shell()
    

    print('\nblock #: ' + str(block_num) + '\ncompute #: ' + str(compute_iter) + '\n')

    [theo_placebo_arm_hists, theo_drug_arm_hists, 
     RR50_emp_stat_powers,   MPC_emp_stat_powers] = \
         generate_theo_pop_hists_and_powers(monthly_mean_lower_bound,
                                            monthly_mean_upper_bound,
                                            monthly_std_dev_lower_bound,
                                            monthly_std_dev_upper_bound,
                                            num_theo_patients_per_trial_arm,
                                            num_baseline_months,
                                            num_testing_months,
                                            minimum_required_baseline_seizure_count,
                                            placebo_mu,
                                            placebo_sigma,
                                            drug_mu,
                                            drug_sigma,
                                            num_trials_per_pop,
                                            num_pops)
    
    store_theo_pop_hists_and_emp_stat_powers(theo_placebo_arm_hists,
                                             theo_drug_arm_hists,
                                             RR50_emp_stat_powers,
                                             MPC_emp_stat_powers,
                                             data_storage_super_folder_path,
                                             block_generic_folder_name,
                                             compute_iter,
                                             block_num)
