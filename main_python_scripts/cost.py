import sys
import os
import json
import numpy as np
import keras.models as models
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary
from utility_code.patient_population_generation import convert_theo_pop_hist

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


    generic_stat_power_model_file_name = 'stat_power_model'

    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'
    stat_power_model = models.load_model(stat_power_model_file_path)

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []
    keras_formatted_theo_placebo_arm_pop_hist = np.zeros((1, 16, 16, 1))
    keras_formatted_theo_drug_arm_pop_hist    = np.zeros((1, 16, 16, 1))

    reached_target_stat_power = False
    placebo_arm_or_drug_arm = 'placebo'
    num_patients = 0

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

            num_patients = num_patients + 1
        
            placebo_arm_init_complete = len(theo_placebo_arm_patient_pop_params_hat_list) != 0
            drug_arm_init_complete    = len(theo_drug_arm_patient_pop_params_hat_list) != 0
            trial_init_complete       = placebo_arm_init_complete and drug_arm_init_complete

            if(placebo_arm_init_complete == True):

                theo_placebo_arm_pop_hist = \
                    convert_theo_pop_hist(1, 16, 1, 16,
                                          np.array(theo_placebo_arm_patient_pop_params_hat_list))
                theo_placebo_arm_pop_hist = 153*( theo_placebo_arm_pop_hist/np.sum(np.sum(theo_placebo_arm_pop_hist, 0)) )
                keras_formatted_theo_placebo_arm_pop_hist[0, :, :, 0] = theo_placebo_arm_pop_hist
    
            if(drug_arm_init_complete == True):

                theo_drug_arm_pop_hist = \
                    convert_theo_pop_hist(1, 16, 1, 16,
                                      np.array(theo_drug_arm_patient_pop_params_hat_list))
                theo_drug_arm_pop_hist = 153*( theo_drug_arm_pop_hist/np.sum(np.sum(theo_drug_arm_pop_hist, 0)) )
                keras_formatted_theo_drug_arm_pop_hist[0, :, :, 0] = theo_drug_arm_pop_hist
        
            if(trial_init_complete == True):

                nn_power = np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_pop_hist, keras_formatted_theo_drug_arm_pop_hist]))

                if(nn_power > 0.9):

                    reached_target_stat_power = True
            
                import pandas as pd
                data_str = '\n' + str([np.round(100*nn_power, 3), num_patients, SNR_hat]) + \
                           '\nplacebo arm:\n' + pd.DataFrame(theo_placebo_arm_pop_hist, index = np.flip(np.arange(1, 17), 0), columns=np.arange(1, 17)).to_string() + \
                           '\ndrug arm:\n'    + pd.DataFrame(theo_drug_arm_pop_hist,    index = np.flip(np.arange(1, 17), 0), columns=np.arange(1, 17)).to_string() + '\n'
                           #'\nSNR map:\n'     + pd.DataFrame(SNR_map >= 0, index = np.flip(np.arange(1, 16), 0), columns=np.arange(1, 16)).to_string() + '\n'
                print(data_str)
            
    
