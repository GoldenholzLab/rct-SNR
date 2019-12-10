import numpy as np
#import time
from .patient_population_generation import randomly_select_theo_patient_pop
from .patient_population_generation import generate_theo_patient_pop_params
from .patient_population_generation import generate_heterogeneous_placebo_arm_patient_pop
from .patient_population_generation import generate_heterogeneous_drug_arm_patient_pop
from .endpoint_functions import calculate_percent_changes
from .endpoint_functions import calculate_time_to_prerandomizations
from .endpoint_functions import calculate_fisher_exact_p_value
from .endpoint_functions import calculate_Mann_Whitney_U_p_value
from .endpoint_functions import calculate_logrank_p_value


def empirically_estimate_RR50_statistical_power(theo_placebo_arm_patient_pop_params,
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

    RR50_p_values = np.zeros(num_trials)
    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 1

    for trial_index in range(num_trials):
        
        [placebo_arm_baseline_seizure_diaries, 
         placebo_arm_testing_seizure_diaries  ] = \
             generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                           theo_placebo_arm_patient_pop_params,
                                                           num_baseline_months,
                                                           num_testing_months,
                                                           baseline_time_scaling_const,
                                                           testing_time_scaling_const,
                                                           minimum_required_baseline_seizure_count,
                                                           placebo_mu,
                                                           placebo_sigma)
    
        [drug_arm_baseline_seizure_diaries, 
         drug_arm_testing_seizure_diaries  ] = \
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

        placebo_arm_percent_changes = \
            calculate_percent_changes(placebo_arm_baseline_seizure_diaries,
                                      placebo_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
        drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_seizure_diaries,
                                      drug_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)

        RR50_p_value = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)
        
        RR50_p_values[trial_index] = RR50_p_value
    
    RR50_emp_stat_power = np.sum(RR50_p_values < 0.05)/num_trials

    return RR50_emp_stat_power


def empirically_estimate_MPC_statistical_power(theo_placebo_arm_patient_pop_params,
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
    
    MPC_p_values = np.zeros(num_trials)
    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 1

    for trial_index in range(num_trials):
        
        [placebo_arm_baseline_seizure_diaries, 
         placebo_arm_testing_seizure_diaries  ] = \
             generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                           theo_placebo_arm_patient_pop_params,
                                                           num_baseline_months,
                                                           num_testing_months,
                                                           baseline_time_scaling_const,
                                                           testing_time_scaling_const,
                                                           minimum_required_baseline_seizure_count,
                                                           placebo_mu,
                                                           placebo_sigma)
    
        [drug_arm_baseline_seizure_diaries, 
         drug_arm_testing_seizure_diaries  ] = \
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

        placebo_arm_percent_changes = \
            calculate_percent_changes(placebo_arm_baseline_seizure_diaries,
                                      placebo_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
        drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_seizure_diaries,
                                      drug_arm_testing_seizure_diaries,
                                      num_theo_patients_per_trial_arm)

        MPC_p_value = \
            calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                             drug_arm_percent_changes)

        MPC_p_values[trial_index] = MPC_p_value
    
    MPC_emp_stat_power = np.sum(MPC_p_values < 0.05)/num_trials

    return MPC_emp_stat_power


def empirically_estimate_TTP_statistical_power(theo_placebo_arm_patient_pop_params,
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

    TTP_p_values = np.zeros(num_trials)
    baseline_time_scaling_const = 1
    testing_time_scaling_const = 28
    num_testing_days = num_testing_months*testing_time_scaling_const

    for trial_index in range(num_trials):

        [placebo_arm_monthly_baseline_seizure_diaries, 
         placebo_arm_daily_testing_seizure_diaries    ] = \
             generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                           theo_placebo_arm_patient_pop_params,
                                                           num_baseline_months,
                                                           num_testing_months,
                                                           baseline_time_scaling_const,
                                                           testing_time_scaling_const,
                                                           minimum_required_baseline_seizure_count,
                                                           placebo_mu,
                                                           placebo_sigma)
    
        [drug_arm_monthly_baseline_seizure_diaries, 
         drug_arm_daily_testing_seizure_diaries    ] = \
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

        [placebo_arm_TTP_times, placebo_arm_observed_array] = \
            calculate_time_to_prerandomizations(placebo_arm_monthly_baseline_seizure_diaries,
                                                placebo_arm_daily_testing_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
        [drug_arm_TTP_times, drug_arm_observed_array] = \
            calculate_time_to_prerandomizations(drug_arm_monthly_baseline_seizure_diaries,
                                                drug_arm_daily_testing_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
        TTP_p_value = \
            calculate_logrank_p_value(placebo_arm_TTP_times, 
                                      placebo_arm_observed_array, 
                                      drug_arm_TTP_times, 
                                      drug_arm_observed_array)
        
        TTP_p_values[trial_index] = TTP_p_value
    
    TTP_emp_stat_power = np.sum(TTP_p_values < 0.05)/num_trials

    return TTP_emp_stat_power

