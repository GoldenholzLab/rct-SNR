import numpy as np
import json
import keras.models as models
import copy
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary_with_minimum_count
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
                         num_baseline_months,
                         minimum_required_baseline_seizure_count):

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


def calculate_SNR(monthly_mean_min,
                  monthly_mean_max,
                  monthly_std_dev_min,
                  monthly_std_dev_max,
                  current_trial_arm,
                  num_SNR_iter,
                  monthly_mean_hat, 
                  monthly_std_dev_hat,
                  theo_placebo_arm_patient_pop_params_hat_list,
                  theo_drug_arm_patient_pop_params_hat_list,
                  stat_power_model):

    num_monthly_std_devs = monthly_std_dev_max - monthly_std_dev_min + 1
    num_monthly_means    = monthly_mean_max    - monthly_mean_min    + 1

    keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist     = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_theo_drug_arm_patient_pop_params_hat_hist        = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat_hist = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat_hist    = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))

    tmp_theo_placebo_arm_patient_pop_params_hat_list = \
        copy.deepcopy(theo_placebo_arm_patient_pop_params_hat_list)
    tmp_theo_drug_arm_patient_pop_params_hat_list = \
        copy.deepcopy(theo_placebo_arm_patient_pop_params_hat_list)

    if( current_trial_arm == 'placebo' ):
        for index in range(num_SNR_iter):
            tmp_theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
    elif( current_trial_arm == 'drug' ):
        for index in range(num_SNR_iter):
            tmp_theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])

    keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(theo_placebo_arm_patient_pop_params_hat_list))

    keras_formatted_theo_drug_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(theo_drug_arm_patient_pop_params_hat_list))

    keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(tmp_theo_placebo_arm_patient_pop_params_hat_list))

    keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat_hist[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(tmp_theo_drug_arm_patient_pop_params_hat_list))

    NN_stat_power = \
        np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat_hist, 
                                             keras_formatted_theo_drug_arm_patient_pop_params_hat_hist]))

    NN_stat_power_w_extra_patient = \
        np.squeeze(stat_power_model.predict([keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat_hist, 
                                             keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat_hist]))

    SNR = NN_stat_power_w_extra_patient - NN_stat_power

    return [SNR, NN_stat_power_w_extra_patient]


def smart_algorithm(monthly_mean_min,
                    monthly_mean_max,
                    monthly_std_dev_min,
                    monthly_std_dev_max,
                    num_baseline_months,
                    minimum_required_baseline_seizure_count,
                    target_stat_power,
                    num_SNR_iter,
                    stat_power_model):

    current_trial_arm = 'placebo'

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    theo_placebo_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                             monthly_mean_max - 1,
                                                                             monthly_std_dev_min,
                                                                             monthly_std_dev_max - 1,
                                                                             num_baseline_months,
                                                                             minimum_required_baseline_seizure_count))

    theo_drug_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                          monthly_mean_max - 1,
                                                                          monthly_std_dev_min,
                                                                          monthly_std_dev_max - 1,
                                                                          num_baseline_months,
                                                                          minimum_required_baseline_seizure_count))

    num_patients = 2
    current_NN_stat_power = 0

    while(current_NN_stat_power < target_stat_power):
    
        accepted_patient = False

        while(not accepted_patient):

            [monthly_mean_hat, 
             monthly_std_dev_hat] = \
                 estimate_patient_loc(monthly_mean_min,
                                      monthly_mean_max - 1,
                                      monthly_std_dev_min,
                                      monthly_std_dev_max - 1,
                                      num_baseline_months,
                                      minimum_required_baseline_seizure_count)

            [SNR, current_NN_stat_power] = \
                calculate_SNR(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              current_trial_arm,
                              num_SNR_iter,
                              monthly_mean_hat, 
                              monthly_std_dev_hat,
                              theo_placebo_arm_patient_pop_params_hat_list,
                              theo_drug_arm_patient_pop_params_hat_list,
                              stat_power_model)

            if(SNR > 0):

                if( current_trial_arm == 'placebo' ):
                    theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                    current_trial_arm == 'drug' 

                elif( current_trial_arm == 'drug' ):
                    theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                    current_trial_arm == 'placebo' 

                num_patients =  num_patients + 1
            
                accepted_patient = True

            #print(accepted_patient)
    
        #print([np.round(100*current_NN_stat_power, 3), num_patients])

    return num_patients


def dumb_algorithm(monthly_mean_min,
                   monthly_mean_max,
                   monthly_std_dev_min,
                   monthly_std_dev_max,
                   num_baseline_months,
                   minimum_required_baseline_seizure_count,
                   target_stat_power,
                   num_SNR_iter,
                   stat_power_model):

    current_trial_arm = 'placebo'

    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    theo_placebo_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                             monthly_mean_max - 1,
                                                                             monthly_std_dev_min,
                                                                             monthly_std_dev_max - 1,
                                                                             num_baseline_months,
                                                                             minimum_required_baseline_seizure_count))

    theo_drug_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                          monthly_mean_max - 1,
                                                                          monthly_std_dev_min,
                                                                          monthly_std_dev_max - 1,
                                                                          num_baseline_months,
                                                                          minimum_required_baseline_seizure_count))

    num_patients = 2
    current_NN_stat_power = 0

    while(current_NN_stat_power < target_stat_power):

        [monthly_mean_hat, 
         monthly_std_dev_hat] = \
             estimate_patient_loc(monthly_mean_min,
                                  monthly_mean_max - 1,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max - 1,
                                  num_baseline_months,
                                  minimum_required_baseline_seizure_count)

        [_, current_NN_stat_power] = \
            calculate_SNR(monthly_mean_min,
                          monthly_mean_max,
                          monthly_std_dev_min,
                          monthly_std_dev_max,
                          current_trial_arm,
                          num_SNR_iter,
                          monthly_mean_hat, 
                          monthly_std_dev_hat,
                          theo_placebo_arm_patient_pop_params_hat_list,
                          theo_drug_arm_patient_pop_params_hat_list,
                          stat_power_model)

        if( current_trial_arm == 'placebo' ):
            theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            current_trial_arm == 'drug' 

        elif( current_trial_arm == 'drug' ):
            theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            current_trial_arm == 'placebo' 

        num_patients =  num_patients + 1
    
        #print([np.round(100*current_NN_stat_power, 3), num_patients])

    return num_patients


if(__name__=='__main__'):

    monthly_mean_min    = 2
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_baseline_months = 2
    minimum_required_baseline_seizure_count = 4

    endpoint_name = 'RR50'
    generic_stat_power_model_file_name = 'stat_power_model'
    target_stat_power = 0.9
    num_SNR_iter = 100
    num_recruitments_processes_for_trials = 100

    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'
    stat_power_model = models.load_model(stat_power_model_file_path)

    num_smart_algo_patients_array = np.zeros(num_recruitments_processes_for_trials)
    num_dumb_algo_patients_array  = np.zeros(num_recruitments_processes_for_trials)

    for recruitment_process_index in range(num_recruitments_processes_for_trials):

        num_smart_algo_patients = \
            smart_algorithm(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max,
                            num_baseline_months,
                            minimum_required_baseline_seizure_count,
                            target_stat_power,
                            num_SNR_iter,
                            stat_power_model)
        
        num_smart_algo_patients_array[recruitment_process_index] = num_smart_algo_patients
    
        num_dumb_algo_patients = \
            dumb_algorithm(monthly_mean_min,
                           monthly_mean_max,
                           monthly_std_dev_min,
                           monthly_std_dev_max,
                           num_baseline_months,
                           minimum_required_baseline_seizure_count,
                           target_stat_power,
                           num_SNR_iter,
                           stat_power_model)
        
        num_dumb_algo_patients_array[recruitment_process_index] = num_dumb_algo_patients

        print('\nrecruitment process #' + str(recruitment_process_index + 1) + '\n')
    
    expected_num_smart_algo_patients = np.round(np.mean(num_smart_algo_patients_array))
    expected_num_dumb_algo_patients  = np.round(np.mean(num_dumb_algo_patients_array))

    print([expected_num_smart_algo_patients, expected_num_dumb_algo_patients])
