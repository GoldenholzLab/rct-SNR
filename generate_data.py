from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import convert_theo_pop_hist
from utility_code.empirical_estimation import empirically_estimate_RR50_statistical_power
import time
import numpy as np


def generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_min,
                                                  monthly_mean_max,
                                                  monthly_std_dev_min,
                                                  monthly_std_dev_max,
                                                  num_theo_patients_per_trial_arm):

    theo_trial_arm_patient_pop_params = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    theo_trial_arm_pop_hist = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              theo_trial_arm_patient_pop_params)
    
    return [theo_trial_arm_patient_pop_params, theo_trial_arm_pop_hist]


def estimate_statistical_power_per_placebo_and_drug_theo_pops(theo_placebo_arm_patient_pop_params,
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

    RR50_emp_stat_power = \
        empirically_estimate_RR50_statistical_power(theo_placebo_arm_patient_pop_params,
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
    
    empirical_estimation_stop_time_in_seconds = time.time()
    empirical_estimation_total_runtime_in_seconds = empirical_estimation_stop_time_in_seconds - empirical_estimation_start_time_in_seconds
    empirical_estimation_total_runtime_in_minutes = empirical_estimation_total_runtime_in_seconds/60
    empirical_estimation_total_runtime_in_minutes_str = 'empirical estimation runtime: ' + str(np.round(empirical_estimation_total_runtime_in_minutes , 3)) + ' minutes'
    print('\n' + empirical_estimation_total_runtime_in_minutes_str + '\n' + RR50_emp_stat_power_str + '\n')

    return RR50_emp_stat_power


def generate_theo_pop_hists_and_power(monthly_mean_min,
                                      monthly_mean_max,
                                      monthly_std_dev_min,
                                      monthly_std_dev_max,
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
     generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_min,
                                                   monthly_mean_max,
                                                   monthly_std_dev_min,
                                                   monthly_std_dev_max,
                                                   num_theo_patients_per_trial_arm)
    
    [theo_drug_arm_patient_pop_params, 
     theo_drug_arm_pop_hist            ] = \
     generate_theo_trial_arm_patient_pops_and_hist(monthly_mean_min,
                                                   monthly_mean_max,
                                                   monthly_std_dev_min,
                                                   monthly_std_dev_max,
                                                   num_theo_patients_per_trial_arm)

    RR50_emp_stat_power = \
        estimate_statistical_power_per_placebo_and_drug_theo_pops(theo_placebo_arm_patient_pop_params,
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
    
    return [theo_placebo_arm_pop_hist, theo_drug_arm_pop_hist, RR50_emp_stat_power]


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_theo_patients_per_trial_arm = 153
    num_baseline_months = 2
    num_testing_months  = 3
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials = 10000
    
    [theo_placebo_arm_pop_hist, 
     theo_drug_arm_pop_hist, 
     RR50_emp_stat_power] = \
         generate_theo_pop_hists_and_power(monthly_mean_min,
                                           monthly_mean_max,
                                           monthly_std_dev_min,
                                           monthly_std_dev_max,
                                           num_theo_patients_per_trial_arm,
                                           num_baseline_months,
                                           num_testing_months,
                                           minimum_required_baseline_seizure_count,
                                           placebo_mu,
                                           placebo_sigma,
                                           drug_mu,
                                           drug_sigma,
                                           num_trials)
    
    
    