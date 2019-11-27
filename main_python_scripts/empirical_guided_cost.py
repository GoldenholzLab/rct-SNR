import numpy as np
import json
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary_with_minimum_count
from utility_code.empirical_estimation import empirically_estimate_all_endpoint_statistical_powers

def retrieve_SNR_map(endpoint_name):

    with open(endpoint_name + '_SNR_data.json', 'r') as json_file:

        SNR_map = np.array(json.load(json_file))
    
    return SNR_map


def get_patient_loc(monthly_mean_min,
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

    estims_within_SNR_map = False

    while(estims_within_SNR_map == False):

        [monthly_mean, monthly_std_dev] = \
            get_patient_loc(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max)

        baseline_monthly_seizure_diary = \
            generate_seizure_diary_with_minimum_count(num_baseline_months, 
                                                      monthly_mean, 
                                                      monthly_std_dev, 
                                                      1,
                                                      minimum_required_baseline_seizure_count)
    
        monthly_mean_hat    = np.int_(np.round(np.mean(baseline_monthly_seizure_diary)))
        monthly_std_dev_hat = np.int_(np.round(np.std(baseline_monthly_seizure_diary)))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )

        estimated_location_overdispersed = monthly_std_dev_hat > np.sqrt(monthly_mean_hat)

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and estimated_location_overdispersed
    
    return [monthly_mean_hat, monthly_std_dev_hat]


if(__name__=='__main__'):

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials_per_stat_power_estim = 1000
    target_stat_power  = 0.9
    num_extra_patients = 20

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    
    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    theo_placebo_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                             monthly_mean_max,
                                                                             monthly_std_dev_min,
                                                                             monthly_std_dev_max,
                                                                             num_baseline_months,
                                                                             minimum_required_baseline_seizure_count))

    theo_drug_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                          monthly_mean_max,
                                                                          monthly_std_dev_min,
                                                                          monthly_std_dev_max,
                                                                          num_baseline_months,
                                                                          minimum_required_baseline_seizure_count))

    num_theo_patients_per_placebo_arm = 1
    num_theo_patients_per_drug_arm = 1
    RR50_stat_power = 0
    MPC_stat_power  = 0
    TTP_stat_power  = 0
    current_trial_arm = 'placebo'

    num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm
    
    underpowered_RR50 = RR50_stat_power < target_stat_power
    underpowered_MPC  = MPC_stat_power  < target_stat_power
    underpowered_TTP  = TTP_stat_power  < target_stat_power
    underpowered_endpoints = underpowered_RR50 or underpowered_MPC or underpowered_TTP

    num_RR50_patients = num_patients
    num_MPC_patients  = num_patients
    num_TTP_patients  = num_patients

    while(underpowered_endpoints):

        [monthly_mean_hat, 
         monthly_std_dev_hat] = \
             estimate_patient_loc(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  num_baseline_months,
                                  minimum_required_baseline_seizure_count)
        
        if( current_trial_arm == 'placebo' ):
            theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            current_trial_arm == 'drug' 
            num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1

        elif( current_trial_arm == 'drug' ):
            theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            current_trial_arm == 'placebo'
            num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1

        num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

        [RR50_stat_power, 
         MPC_stat_power, 
         TTP_stat_power] = \
             empirically_estimate_all_endpoint_statistical_powers(num_theo_patients_per_placebo_arm,
                                                                  num_theo_patients_per_drug_arm,
                                                                  np.array(theo_placebo_arm_patient_pop_params_hat_list),
                                                                  np.array(theo_drug_arm_patient_pop_params_hat_list),
                                                                  num_baseline_months,
                                                                  num_testing_months,
                                                                  minimum_required_baseline_seizure_count,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  drug_mu,
                                                                  drug_sigma,
                                                                  num_trials_per_stat_power_estim)
        
        underpowered_RR50 = RR50_stat_power < target_stat_power
        underpowered_MPC  = MPC_stat_power  < target_stat_power
        underpowered_TTP  = TTP_stat_power  < target_stat_power

        if(underpowered_RR50):
            num_RR50_patients = num_patients
        if(underpowered_MPC):
            num_MPC_patients = num_patients
        if(underpowered_TTP):
            num_TTP_patients = num_patients

        print()
        print([RR50_stat_power,   MPC_stat_power,   TTP_stat_power])
        print([num_RR50_patients, num_MPC_patients, num_TTP_patients])
        print()

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
