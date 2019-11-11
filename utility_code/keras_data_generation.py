import numpy as np
from patient_population_generation import generate_theo_patient_pop_params
from patient_population_generation import generate_heterogenous_placebo_arm_patient_pop
from patient_population_generation import generate_heterogenous_drug_arm_patient_pop
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


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    max_theo_patients_per_trial_arm  = 50
    theo_patients_per_trial_arm_step = 10

    num_baseline_months = 2
    num_testing_months  = 3
    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 28
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials = 5

    [theo_placebo_arm_patient_pop_params, 
     theo_drug_arm_patient_pop_params] = \
         generate_theo_patient_pop_params_per_trial_arm(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        max_theo_patients_per_trial_arm)

    patient_nums = np.arange(theo_patients_per_trial_arm_step, max_theo_patients_per_trial_arm + theo_patients_per_trial_arm_step, theo_patients_per_trial_arm_step)

    RR50_p_value_matrix = np.zeros((np.int_(max_theo_patients_per_trial_arm/theo_patients_per_trial_arm_step), num_trials))

    for trial_index in range(num_trials):

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
        
        for patient_num in patient_nums:

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

            RR50_p_value_matrix[patient_num - 1, trial_index] = RR50_p_value

    import pandas as pd
    print(pd.DataFrame(RR50_p_value_matrix).to_string())
