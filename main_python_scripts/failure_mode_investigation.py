import numpy as np
import sys
import os
import time
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import generate_heterogenous_placebo_arm_patient_pop
from utility_code.patient_population_generation import generate_heterogenous_drug_arm_patient_pop
from utility_code.endpoint_functions import calculate_percent_changes
from utility_code.endpoint_functions import calculate_fisher_exact_p_value
from utility_code.endpoint_functions import calculate_Mann_Whitney_U_p_value


def generate_patient_populations(monthly_mean_min,
                                 monthly_mean_max,
                                 monthly_std_dev_min,
                                 monthly_std_dev_max,
                                 num_theo_patients_per_trial_arm,
                                 num_theo_patients_per_trial_arm_in_loc,
                                 local_monthly_mean,
                                 local_monthly_std_dev,
                                 loc_in_placebo_or_drug):

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
    
    theo_placebo_arm_patient_pop_params_with_loc = np.copy(theo_placebo_arm_patient_pop_params)
    theo_drug_arm_patient_pop_params_with_loc    = np.copy(theo_drug_arm_patient_pop_params)
    patient_pop_params_per_one_loc = np.transpose(np.vstack([np.full(num_theo_patients_per_trial_arm_in_loc, local_monthly_mean),
                                                             np.full(num_theo_patients_per_trial_arm_in_loc, local_monthly_std_dev)]))

    if(loc_in_placebo_or_drug == 'placebo'):
        theo_placebo_arm_patient_pop_params_with_loc = np.vstack([theo_placebo_arm_patient_pop_params_with_loc, patient_pop_params_per_one_loc])
    elif(loc_in_placebo_or_drug == 'drug'):
        theo_drug_arm_patient_pop_params_with_loc    = np.vstack([theo_drug_arm_patient_pop_params_with_loc,    patient_pop_params_per_one_loc])

    return [theo_placebo_arm_patient_pop_params,
            theo_drug_arm_patient_pop_params,
            theo_placebo_arm_patient_pop_params_with_loc,
            theo_drug_arm_patient_pop_params_with_loc]


def estimate_statistical_power_of_pop(num_theo_patients_per_trial_arm,
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
                                      drug_sigma,
                                      num_trials):

    RR50_p_value_array = np.zeros(num_trials)
    MPC_p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):
    
        [placebo_arm_baseline_seizure_diaries, 
         placebo_arm_testing_seizure_diaries  ] = \
             generate_heterogenous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                           theo_placebo_arm_patient_pop_params,
                                                           num_baseline_months,
                                                           num_testing_months,
                                                           baseline_time_scaling_const,
                                                           testing_time_scaling_const,
                                                           minimum_required_baseline_seizure_count,
                                                           placebo_mu,
                                                           placebo_sigma)
    
        placebo_arm_percent_changes = \
            calculate_percent_changes(placebo_arm_baseline_seizure_diaries,
                                      placebo_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
        [drug_arm_baseline_seizure_diaries, 
         drug_arm_testing_seizure_diaries  ] = \
             generate_heterogenous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
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
    
        drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_seizure_diaries,
                                      drug_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
        RR50_p_value_array[trial_index] = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)
        
        MPC_p_value_array[trial_index] = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)
    
    RR50_stat_power = np.sum(RR50_p_value_array < 0.05)/num_trials
    MPC_stat_power  = np.sum(MPC_p_value_array < 0.05)/num_trials

    return [RR50_stat_power, MPC_stat_power]


def estimate_statistical_power_increases_at_loc(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                num_theo_patients_per_trial_arm,
                                                num_baseline_months,
                                                num_testing_months,
                                                baseline_time_scaling_const,
                                                testing_time_scaling_const,
                                                minimum_required_baseline_seizure_count,
                                                placebo_mu,
                                                placebo_sigma,
                                                drug_mu,
                                                drug_sigma,
                                                num_trials,
                                                num_theo_patients_per_trial_arm_in_loc,
                                                local_monthly_mean,
                                                local_monthly_std_dev,
                                                num_stat_power_increase_estimates_per_trial_arm,
                                                loc_in_placebo_or_drug):

    RR50_stat_power_increases_array = np.zeros(num_stat_power_increase_estimates_per_trial_arm)
    MPC_stat_power_increases_array  = np.zeros(num_stat_power_increase_estimates_per_trial_arm)

    for stat_power_increase_estimate_index in range(num_stat_power_increase_estimates_per_trial_arm):

        start_time_in_seconds = time.time()

        [theo_placebo_arm_patient_pop_params,
         theo_drug_arm_patient_pop_params,
         theo_placebo_arm_patient_pop_params_with_loc,
         theo_drug_arm_patient_pop_params_with_loc] = \
             generate_patient_populations(monthly_mean_min,
                                          monthly_mean_max,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max,
                                          num_theo_patients_per_trial_arm,
                                          num_theo_patients_per_trial_arm_in_loc,
                                          local_monthly_mean,
                                          local_monthly_std_dev,
                                          loc_in_placebo_or_drug)
    
        [RR50_stat_power, MPC_stat_power] = \
            estimate_statistical_power_of_pop(num_theo_patients_per_trial_arm,
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
                                              drug_sigma,
                                              num_trials)

        [RR50_stat_power_with_loc, MPC_stat_power_with_loc] = \
            estimate_statistical_power_of_pop(num_theo_patients_per_trial_arm,
                                              theo_placebo_arm_patient_pop_params_with_loc,
                                              theo_drug_arm_patient_pop_params_with_loc,
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

        RR50_stat_power_increases_array[stat_power_increase_estimate_index] = RR50_stat_power_with_loc - RR50_stat_power
        MPC_stat_power_increases_array[stat_power_increase_estimate_index]  = MPC_stat_power_with_loc  - MPC_stat_power

        stop_time_in_seconds = time.time()
        total_runtime_in_seconds = stop_time_in_seconds - start_time_in_seconds
        total_runtime_in_minutes = total_runtime_in_seconds/60
        total_runtime_in_minutes_str = str(np.round(total_runtime_in_minutes, 3))

        print(loc_in_placebo_or_drug + ' #' + str(stat_power_increase_estimate_index + 1) + ': ' + total_runtime_in_minutes_str + ' minutes')
    
    return [RR50_stat_power_increases_array, 
            MPC_stat_power_increases_array]


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 15
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_baseline_months = 2
    num_testing_months  = 3

    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 1

    minimum_required_baseline_seizure_count = 4
    
    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05
    
    num_theo_patients_per_trial_arm = 50

    num_trials = 500

    local_monthly_mean = 1
    local_monthly_std_dev = 13
    num_theo_patients_per_trial_arm_in_loc = 20

    num_stat_power_increase_estimates_per_trial_arm = 50
    loc_in_placebo_or_drug = 'placebo'

    [placebo_arm_RR50_stat_power_increases_array, 
     placebo_arm_MPC_stat_power_increases_array] = \
        estimate_statistical_power_increases_at_loc(monthly_mean_min,
                                                    monthly_mean_max,
                                                    monthly_std_dev_min,
                                                    monthly_std_dev_max,
                                                    num_theo_patients_per_trial_arm,
                                                    num_baseline_months,
                                                    num_testing_months,
                                                    baseline_time_scaling_const,
                                                    testing_time_scaling_const,
                                                    minimum_required_baseline_seizure_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma,
                                                    num_trials,
                                                    num_theo_patients_per_trial_arm_in_loc,
                                                    local_monthly_mean,
                                                    local_monthly_std_dev,
                                                    num_stat_power_increase_estimates_per_trial_arm,
                                                    'placebo')
    
    [drug_arm_RR50_stat_power_increases_array, 
     drug_arm_MPC_stat_power_increases_array] = \
        estimate_statistical_power_increases_at_loc(monthly_mean_min,
                                                    monthly_mean_max,
                                                    monthly_std_dev_min,
                                                    monthly_std_dev_max,
                                                    num_theo_patients_per_trial_arm,
                                                    num_baseline_months,
                                                    num_testing_months,
                                                    baseline_time_scaling_const,
                                                    testing_time_scaling_const,
                                                    minimum_required_baseline_seizure_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma,
                                                    num_trials,
                                                    num_theo_patients_per_trial_arm_in_loc,
                                                    local_monthly_mean,
                                                    local_monthly_std_dev,
                                                    num_stat_power_increase_estimates_per_trial_arm,
                                                    'drug')
    
    RR50_stat_power_increases_array = np.concatenate([placebo_arm_RR50_stat_power_increases_array, drug_arm_RR50_stat_power_increases_array], 0)
    MPC_stat_power_increases_array  = np.concatenate([placebo_arm_MPC_stat_power_increases_array,  drug_arm_MPC_stat_power_increases_array],  0)

    RR50_stat_power_increase_at_loc = np.mean(RR50_stat_power_increases_array)
    MPC_stat_power_increase_at_loc  = np.mean(MPC_stat_power_increases_array)

    print('\nRR50 statistical power increase: ' + str(np.round(RR50_stat_power_increase_at_loc, 3)) + ' %\n')
    print('\nMPC statistical power increase:  ' + str(np.round(MPC_stat_power_increase_at_loc, 3))  + ' %\n')
