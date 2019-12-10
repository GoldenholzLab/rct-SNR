import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_baseline_seizure_diary
from utility_code.patient_population_generation import generate_heterogeneous_placebo_arm_patient_pop
from utility_code.patient_population_generation import generate_heterogeneous_drug_arm_patient_pop


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


def add_patient_to_placebo_theo_patient_pop(trial_arm_theo_patient_pop_list):

    [monthly_mean_hat, 
     monthly_std_dev_hat] = \
         estimate_patient_loc(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              minimum_required_baseline_seizure_count)
    
    trial_arm_theo_patient_pop_list.append([monthly_mean_hat, monthly_std_dev_hat])

    return trial_arm_theo_patient_pop_list


if(__name__=='__main__'):

    # SNR map parameters
    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    # RCT design parameters
    endpoint_name = 'RR50'
    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4

    # simulation parameters
    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#

    placebo_arm_theo_patient_pop_list = []
    drug_arm_theo_patient_pop_list    = []

    placebo_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(placebo_arm_theo_patient_pop_list)

    drug_arm_theo_patient_pop_list = \
        add_patient_to_placebo_theo_patient_pop(drug_arm_theo_patient_pop_list)

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#

    baseline_time_scaling_const = 1
    testing_time_scaling_const  = 1

    num_theo_patients_in_placebo_arm = len(placebo_arm_theo_patient_pop_list)
    num_theo_patients_in_drug_arm = len(drug_arm_theo_patient_pop_list)

    theo_placebo_arm_patient_pop_params = np.array(placebo_arm_theo_patient_pop_list)
    theo_drug_arm_patient_pop_params = np.array(drug_arm_theo_patient_pop_list)

    [placebo_arm_baseline_seizure_diaries, 
     placebo_arm_testing_seizure_diaries  ] = \
         generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_in_placebo_arm,
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

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------------#
    