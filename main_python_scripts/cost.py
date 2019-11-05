import sys
import os
import json
import numpy as np
#import keras.models as models
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary
from utility_code.empirical_estimation import empirically_estimate_imbalanced_RR50_statistical_power


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
    realistic = False

    while((not overdispersed) or (not non_zero_mean) or (not realistic)):

        overdispersed = False
        non_zero_mean = False
        realistic = False
        
        monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max    + 1)
        monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max + 1)

        if(monthly_mean != 0):

                non_zero_mean = True

        if(monthly_std_dev > np.sqrt(monthly_mean)):

            overdispersed = True
        
        if(monthly_std_dev < np.power(monthly_mean, 1.3)):

            realistic = True
    
    return [monthly_mean, monthly_std_dev]


def estimate_patient_loc(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_baseline_months):

    '''

    I suspect that standard deviation is conitually underestimated.

    '''

    estims_within_SNR_map = False

    while(estims_within_SNR_map == False):

        [monthly_mean, monthly_std_dev] = \
            get_patient_loc(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max)

        baseline_monthly_seizure_diary = \
            generate_seizure_diary(num_baseline_months, 
                                   monthly_mean, 
                                   monthly_std_dev, 
                                   1)
    
        monthly_mean_hat    = np.int_(np.round(np.mean(baseline_monthly_seizure_diary)))
        monthly_std_dev_hat = np.int_(np.round(np.std(baseline_monthly_seizure_diary)))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )

        estimated_location_overdispersed = monthly_std_dev_hat > np.sqrt(monthly_mean_hat)

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and estimated_location_overdispersed
    
    return [monthly_mean_hat, monthly_std_dev_hat]


if(__name__=='__main__'):

    endpoint_name = sys.argv[1]

    SNR_map = retrieve_SNR_map(endpoint_name)

    monthly_mean_min    = 1
    monthly_mean_max    = 15
    monthly_std_dev_min = 1
    monthly_std_dev_max = 15

    num_baseline_months = 2
    num_testing_months  = 3
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05
    
    num_trials = 2000

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    reached_target_stat_power = False
    placebo_arm_or_drug_arm = 'placebo'
    num_theo_patients = 0

    RR50_emp_stat_power = 0

    while(reached_target_stat_power == False):

        [monthly_mean_hat, monthly_std_dev_hat] = \
            estimate_patient_loc(monthly_mean_min,
                                 monthly_mean_max,
                                 monthly_std_dev_min,
                                 monthly_std_dev_max,
                                 num_baseline_months)
    
        SNR_hat = SNR_map[monthly_std_dev_max - monthly_std_dev_hat, monthly_mean_hat - 1]/20

        if(SNR_hat >= 0):
            
            if(placebo_arm_or_drug_arm == 'placebo'):
                
                theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                placebo_arm_or_drug_arm = 'drug'

            elif(placebo_arm_or_drug_arm == 'drug'):

                theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                placebo_arm_or_drug_arm = 'placebo'

            num_theo_patients = num_theo_patients + 1
        
            placebo_arm_init_complete = len(theo_placebo_arm_patient_pop_params_hat_list) != 0
            drug_arm_init_complete    = len(theo_drug_arm_patient_pop_params_hat_list) != 0
            trial_init_complete       = placebo_arm_init_complete and drug_arm_init_complete
        
            if(trial_init_complete == True):

                RR50_emp_stat_power = \
                    empirically_estimate_imbalanced_RR50_statistical_power(np.array(theo_placebo_arm_patient_pop_params_hat_list),
                                                                           np.array(theo_drug_arm_patient_pop_params_hat_list),
                                                                           len(theo_placebo_arm_patient_pop_params_hat_list),
                                                                           len(theo_drug_arm_patient_pop_params_hat_list),
                                                                           num_baseline_months,
                                                                           num_testing_months,
                                                                           minimum_required_baseline_seizure_count,
                                                                           placebo_mu,
                                                                           placebo_sigma,
                                                                           drug_mu,
                                                                           drug_sigma,
                                                                           num_trials)

                if(RR50_emp_stat_power > 0.9):

                    reached_target_stat_power = True
                
                print( str(np.transpose(np.array(theo_placebo_arm_patient_pop_params_hat_list))) + '\n' + str(np.transpose(np.array(theo_drug_arm_patient_pop_params_hat_list))) )
                print([np.round(100*RR50_emp_stat_power, 3), num_theo_patients])

            
    
