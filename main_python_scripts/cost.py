import numpy as np
import json
import keras.models as models
import sys
import os
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
                         num_baseline_months):

    '''

    I suspect that standard deviation is continually underestimated.

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


def smart_algorithm(monthly_mean_min,
                    monthly_mean_max,
                    monthly_std_dev_min,
                    monthly_std_dev_max,
                    num_baseline_months,
                    SNR_map,
                    stat_power_model,
                    target_stat_power):

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    reached_target_stat_power = False
    placebo_arm_or_drug_arm = 'placebo'
    num_theo_patients = 0

    keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist = np.zeros((1, 16, 16, 1))
    keras_formatted_theo_drug_arm_patient_pop_params_hat_hist    = np.zeros((1, 16, 16, 1))

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
            number_of_patients_is_multiple_of_ten = (num_theo_patients % 10) == 0
        
            if( trial_init_complete and number_of_patients_is_multiple_of_ten ):

                keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
                    convert_theo_pop_hist(monthly_mean_min,
                                          monthly_mean_max + 1,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max + 1,
                                          np.array(theo_placebo_arm_patient_pop_params_hat_list))
                
                keras_formatted_theo_drug_arm_patient_pop_params_hat_hist[0, :, :, 0]  = \
                    convert_theo_pop_hist(monthly_mean_min,
                                          monthly_mean_max + 1,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max + 1,
                                          np.array(theo_drug_arm_patient_pop_params_hat_list))

                NN_stat_power = \
                    np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist, 
                                                         keras_formatted_theo_drug_arm_patient_pop_params_hat_hist    ]))


                if(NN_stat_power >= target_stat_power):

                    reached_target_stat_power = True
                
                
                print('\n' + str(np.round(100*NN_stat_power, 3)) + ' %, ' + 
                             str(num_theo_patients)               + ' smart patients')
                

    return num_theo_patients


def dumb_algorithm(monthly_mean_min,
                   monthly_mean_max,
                   monthly_std_dev_min,
                   monthly_std_dev_max,
                   num_baseline_months,
                   stat_power_model,
                   target_stat_power):

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    reached_target_stat_power = False
    placebo_arm_or_drug_arm = 'placebo'
    num_theo_patients = 0

    keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist = np.zeros((1, 16, 16, 1))
    keras_formatted_theo_drug_arm_patient_pop_params_hat_hist    = np.zeros((1, 16, 16, 1))

    while(reached_target_stat_power == False):

        [monthly_mean_hat, monthly_std_dev_hat] = \
            estimate_patient_loc(monthly_mean_min,
                                 monthly_mean_max,
                                 monthly_std_dev_min,
                                 monthly_std_dev_max,
                                 num_baseline_months)
            
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
        number_of_patients_is_multiple_of_ten = (num_theo_patients % 10) == 0
        
        if( trial_init_complete and number_of_patients_is_multiple_of_ten ):

            keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
                convert_theo_pop_hist(monthly_mean_min,
                                      monthly_mean_max + 1,
                                      monthly_std_dev_min,
                                      monthly_std_dev_max + 1,
                                      np.array(theo_placebo_arm_patient_pop_params_hat_list))
                
            keras_formatted_theo_drug_arm_patient_pop_params_hat_hist[0, :, :, 0]  = \
                convert_theo_pop_hist(monthly_mean_min,
                                      monthly_mean_max + 1,
                                      monthly_std_dev_min,
                                      monthly_std_dev_max + 1,
                                      np.array(theo_drug_arm_patient_pop_params_hat_list))

            NN_stat_power = \
                np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist, 
                                                     keras_formatted_theo_drug_arm_patient_pop_params_hat_hist    ]))


            if(NN_stat_power >= target_stat_power):

                reached_target_stat_power = True

            
            print('\n' + str(np.round(100*NN_stat_power, 3)) + ' %, ' + 
                         str(num_theo_patients)               + ' dumb patients')
            

    return num_theo_patients


if(__name__=='__main__'):

    monthly_mean_min = 1
    monthly_mean_max = 15
    monthly_std_dev_min = 1
    monthly_std_dev_max = 15
    num_baseline_months = 2

    endpoint_name = 'TTP'
    generic_stat_power_model_file_name = 'stat_power_model'
    target_stat_power = 0.9
    num_recruitments_processes_for_trials = 1000

    num_smart_theo_patients_array = np.zeros(num_recruitments_processes_for_trials)
    num_dumb_theo_patients_array  = np.zeros(num_recruitments_processes_for_trials)

    SNR_map = retrieve_SNR_map(endpoint_name)
    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'
    stat_power_model = models.load_model(stat_power_model_file_path)

    for recruitment_process_index in range(num_recruitments_processes_for_trials):

        num_smart_theo_patients = \
            smart_algorithm(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max,
                            num_baseline_months,
                            SNR_map,
                            stat_power_model,
                            target_stat_power)

        num_smart_theo_patients_array[recruitment_process_index] = num_smart_theo_patients

        num_dumb_theo_patients = \
            dumb_algorithm(monthly_mean_min,
                           monthly_mean_max,
                           monthly_std_dev_min,
                           monthly_std_dev_max,
                           num_baseline_months,
                           stat_power_model,
                           target_stat_power)
        
        num_dumb_theo_patients_array[recruitment_process_index] = num_dumb_theo_patients

        print('\nrecruitment process #' + str(recruitment_process_index + 1) + '\n')
    
    expected_num_smart_theo_patients = np.round(np.mean(num_smart_theo_patients_array))
    expected_num_dumb_theo_patients  = np.round(np.mean(num_dumb_theo_patients_array))

    print([expected_num_smart_theo_patients, expected_num_dumb_theo_patients])