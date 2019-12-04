import numpy as np
import json
import keras.models as models
import copy
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary_with_minimum_count
from utility_code.patient_population_generation import convert_theo_pop_hist
from utility_code.empirical_estimation import empirically_estimate_endpoint_statistical_power


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
                                                      28,
                                                      minimum_required_baseline_seizure_count)
    
        monthly_mean_hat    = np.int_(np.round(28*np.mean(baseline_monthly_seizure_diary)))
        monthly_std_dev_hat = np.int_(np.round(np.sqrt(28)*np.std(baseline_monthly_seizure_diary)))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )

        estimated_location_overdispersed = monthly_std_dev_hat > np.sqrt(monthly_mean_hat)

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and estimated_location_overdispersed

    return [monthly_mean_hat, monthly_std_dev_hat]


def calculate_SNR(theo_placebo_arm_patient_pop_params_hat_list,
                  theo_drug_arm_patient_pop_params_hat_list,
                  monthly_mean_hat,
                  monthly_std_dev_hat,
                  current_trial_arm,
                  num_extra_patients,
                  monthly_mean_min,
                  monthly_mean_max,
                  monthly_std_dev_min,
                  monthly_std_dev_max,
                  stat_power_model):

    num_monthly_means    = monthly_mean_max    - monthly_mean_min + 1
    num_monthly_std_devs = monthly_std_dev_max - monthly_std_dev_min + 1

    keras_formatted_theo_placebo_arm_patient_pop_params_hat     = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_theo_drug_arm_patient_pop_params_hat        = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat    = np.zeros((1, num_monthly_std_devs, num_monthly_means, 1))

    tmp_theo_placebo_arm_patient_pop_params_hat_list = \
        copy.deepcopy(theo_placebo_arm_patient_pop_params_hat_list)
        
    tmp_theo_drug_arm_patient_pop_params_hat_list = \
        copy.deepcopy(theo_drug_arm_patient_pop_params_hat_list)

    for patient_index in range(num_extra_patients):
        tmp_theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat,
                                                                 monthly_std_dev_hat])
    for patient_index in range(num_extra_patients):
        tmp_theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat,
                                                              monthly_std_dev_hat])
        
    keras_formatted_theo_placebo_arm_patient_pop_params_hat[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(theo_placebo_arm_patient_pop_params_hat_list))

    keras_formatted_theo_drug_arm_patient_pop_params_hat[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(theo_placebo_arm_patient_pop_params_hat_list))

    keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(tmp_theo_placebo_arm_patient_pop_params_hat_list))

    keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat[0, :, :, 0] = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              np.array(tmp_theo_placebo_arm_patient_pop_params_hat_list))
        
    current_power = \
        np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat, 
                                             keras_formatted_theo_drug_arm_patient_pop_params_hat]))
        
    power_with_extra_patients = \
        np.squeeze(stat_power_model.predict([keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat, 
                                             keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat]))
        
    loc_SNR = power_with_extra_patients - current_power

    postive_SNR = loc_SNR > 0

    return postive_SNR


def dumb_algorithm(monthly_mean_min,
                   monthly_mean_max,
                   monthly_std_dev_min,
                   monthly_std_dev_max,
                   num_baseline_months,
                   num_testing_months,
                   minimum_required_baseline_seizure_count,
                   placebo_mu,
                   placebo_sigma,
                   drug_mu,
                   drug_sigma,
                   num_trials_per_stat_power_estim,
                   target_stat_power,
                   update_RR50,
                   update_MPC,
                   update_TTP):

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
    current_trial_arm = 'placebo'

    underpowered_endpoint = True

    while(underpowered_endpoint):

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
            num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1
            current_trial_arm = 'drug'

        elif( current_trial_arm == 'drug' ):
            theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1
            current_trial_arm = 'placebo'
            
        if(update_RR50 or update_MPC):

            stat_power = \
                empirically_estimate_endpoint_statistical_power(num_theo_patients_per_placebo_arm,
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
                                                                num_trials_per_stat_power_estim,
                                                                update_RR50,
                                                                update_MPC,
                                                                update_TTP)
            
        elif(update_TTP):

            [stat_power,
             average_placebo_TTP,
             average_drug_TTP] = \
                 empirically_estimate_endpoint_statistical_power(num_theo_patients_per_placebo_arm,
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
                                                                 num_trials_per_stat_power_estim,
                                                                 update_RR50,
                                                                 update_MPC,
                                                                 update_TTP)
                
        if( stat_power >= target_stat_power ):

            underpowered_endpoint = False
        
        print('accepted: '           + str([monthly_mean_hat, monthly_std_dev_hat])                            + '\n' + \
              'number of patients: ' + str(num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm) + '\n' + \
               str(underpowered_endpoint) + '\n' + \
              'statistical power: '  + str(np.round(100*stat_power, 3))                                        + ' %\n'  )

    if(update_RR50 or update_MPC):

        num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

        return num_patients
    
    elif(update_TTP):

        return [num_theo_patients_per_placebo_arm,
                num_theo_patients_per_drug_arm,
                average_placebo_TTP,
                average_drug_TTP]


def smart_algorithm(monthly_mean_min,
                    monthly_mean_max,
                    monthly_std_dev_min,
                    monthly_std_dev_max,
                    num_baseline_months,
                    num_testing_months,
                    minimum_required_baseline_seizure_count,
                    placebo_mu,
                    placebo_sigma,
                    drug_mu,
                    drug_sigma,
                    num_trials_per_stat_power_estim,
                    target_stat_power,
                    num_extra_patients,
                    stat_power_model,
                    update_RR50,
                    update_MPC,
                    update_TTP):

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
    current_trial_arm = 'placebo'

    underpowered_endpoint = True

    while(underpowered_endpoint):

        [monthly_mean_hat, 
         monthly_std_dev_hat] = \
             estimate_patient_loc(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  num_baseline_months,
                                  minimum_required_baseline_seizure_count)

        positive_SNR = \
            calculate_SNR(theo_placebo_arm_patient_pop_params_hat_list,
                          theo_drug_arm_patient_pop_params_hat_list,
                          monthly_mean_hat,
                          monthly_std_dev_hat,
                          current_trial_arm,
                          num_extra_patients,
                          monthly_mean_min,
                          monthly_mean_max,
                          monthly_std_dev_min,
                          monthly_std_dev_max,
                          stat_power_model)
        
        if(positive_SNR):

            if( current_trial_arm == 'placebo' ):
                theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1
                current_trial_arm = 'drug'

            elif( current_trial_arm == 'drug' ):
                theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1
                current_trial_arm = 'placebo'
            
            if(update_RR50 or update_MPC):

                stat_power = \
                    empirically_estimate_endpoint_statistical_power(num_theo_patients_per_placebo_arm,
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
                                                                    num_trials_per_stat_power_estim,
                                                                    update_RR50,
                                                                    update_MPC,
                                                                    update_TTP)
            
            elif(update_TTP):

                [stat_power,
                 average_placebo_TTP,
                 average_drug_TTP] = \
                     empirically_estimate_endpoint_statistical_power(num_theo_patients_per_placebo_arm,
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
                                                                     num_trials_per_stat_power_estim,
                                                                     update_RR50,
                                                                     update_MPC,
                                                                     update_TTP)
                
            if( stat_power >= target_stat_power ):

                underpowered_endpoint = False

            print('accepted: '           + str([monthly_mean_hat, monthly_std_dev_hat])                            + '\n' + \
                  'number of patients: ' + str(num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm) + '\n' + \
                   str(underpowered_endpoint)                                                                      + '\n' + \
                   str(positive_SNR)                                                                               + '\n' + \
                  'statistical power: '  + str(np.round(100*stat_power, 3))                                        + ' %\n'  )
        
        else:

            print('\n-----------------------------------------------------------------------------------------------------------\nrejected: ' + str([monthly_mean_hat, monthly_std_dev_hat]))

    if(update_RR50 or update_MPC):

        num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

        return num_patients
    
    elif(update_TTP):

        return [num_theo_patients_per_placebo_arm,
                num_theo_patients_per_drug_arm,
                average_placebo_TTP,
                average_drug_TTP]
    

def save_results(endpoint_name,
                 smart_or_dumb,
                 iter_index,
                 data_list):

    folder_name = os.getcwd() + '/' + endpoint_name + '_' + smart_or_dumb +'_data'

    if( not os.path.isdir(folder_name) ):

        os.makedirs(folder_name)
    
    with open(folder_name + '/' + str(iter_index) + '.json', 'w+') as json_file:
        
        json.dump(data_list, json_file)


def take_inputs_from_command_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    num_baseline_months = int(sys.argv[5])
    num_testing_months =  int(sys.argv[6])
    minimum_required_baseline_seizure_count = int(sys.argv[7])

    placebo_mu    = float(sys.argv[8])
    placebo_sigma = float(sys.argv[9])
    drug_mu       = float(sys.argv[10])
    drug_sigma    = float(sys.argv[11])

    num_trials_per_stat_power_estim = int(sys.argv[12])
    target_stat_power  = float(sys.argv[13])
    num_extra_patients =   int(sys.argv[14])

    generic_stat_power_model_file_name = sys.argv[15]
    smart_or_dumb =     sys.argv[16]
    endpoint_name =     sys.argv[17]
    iter_index    = int(sys.argv[18])

    '''
    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_baseline_months = 2
    num_testing_months  = 3
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials_per_stat_power_estim = 10
    target_stat_power = 0.3
    num_extra_patients = 10

    generic_stat_power_model_file_name='stat_power_model_copy'
    smart_or_dumb =     sys.argv[1]
    endpoint_name =     sys.argv[2]
    iter_index    = int(sys.argv[3])
    '''

    return [monthly_mean_min,
            monthly_mean_max,
            monthly_std_dev_min,
            monthly_std_dev_max,
            num_baseline_months,
            num_testing_months,
            minimum_required_baseline_seizure_count,
            placebo_mu,
            placebo_sigma,
            drug_mu,
            drug_sigma,
            num_trials_per_stat_power_estim,
            target_stat_power,
            num_extra_patients,
            generic_stat_power_model_file_name,
            smart_or_dumb,
            endpoint_name,
            iter_index]


if(__name__=='__main__'):

    [monthly_mean_min,
     monthly_mean_max,
     monthly_std_dev_min,
     monthly_std_dev_max,
     num_baseline_months,
     num_testing_months,
     minimum_required_baseline_seizure_count,
     placebo_mu,
     placebo_sigma,
     drug_mu,
     drug_sigma,
     num_trials_per_stat_power_estim,
     target_stat_power,
     num_extra_patients,
     generic_stat_power_model_file_name,
     smart_or_dumb,
     endpoint_name,
     iter_index] = \
         take_inputs_from_command_shell()

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#

    stat_power_model = models.load_model(endpoint_name + '_' + generic_stat_power_model_file_name + '.h5')

    update_RR50 = endpoint_name == 'RR50'
    update_MPC  = endpoint_name == 'MPC'
    update_TTP  = endpoint_name == 'TTP'

    print(smart_or_dumb + ', iter #' + str(iter_index))

    if(smart_or_dumb == 'dumb'):
    
        if(update_RR50 or update_MPC):

            num_patients = \
                dumb_algorithm(monthly_mean_min,
                               monthly_mean_max,
                               monthly_std_dev_min,
                               monthly_std_dev_max,
                               num_baseline_months,
                               num_testing_months,
                               minimum_required_baseline_seizure_count,
                               placebo_mu,
                               placebo_sigma,
                               drug_mu,
                               drug_sigma,
                               num_trials_per_stat_power_estim,
                               target_stat_power,
                               update_RR50,
                               update_MPC,
                               update_TTP)
            
            data_list = num_patients

        if(update_TTP):

            [num_theo_patients_per_placebo_arm,
             num_theo_patients_per_drug_arm,
             average_placebo_TTP,
             average_drug_TTP] = \
                 dumb_algorithm(monthly_mean_min,
                                monthly_mean_max,
                                monthly_std_dev_min,
                                monthly_std_dev_max,
                                num_baseline_months,
                                num_testing_months,
                                minimum_required_baseline_seizure_count,
                                placebo_mu,
                                placebo_sigma,
                                drug_mu,
                                drug_sigma,
                                num_trials_per_stat_power_estim,
                                target_stat_power,
                                update_RR50,
                                update_MPC,
                                update_TTP)
            
            data_list = [num_theo_patients_per_placebo_arm, 
                         num_theo_patients_per_drug_arm,
                         average_placebo_TTP,
                         average_drug_TTP]

    elif(smart_or_dumb == 'smart'):

        if(update_RR50 or update_MPC):

            num_patients = \
                smart_algorithm(monthly_mean_min,
                                monthly_mean_max,
                                monthly_std_dev_min,
                                monthly_std_dev_max,
                                num_baseline_months,
                                num_testing_months,
                                minimum_required_baseline_seizure_count,
                                placebo_mu,
                                placebo_sigma,
                                drug_mu,
                                drug_sigma,
                                num_trials_per_stat_power_estim,
                                target_stat_power,
                                num_extra_patients,
                                stat_power_model,
                                update_RR50,
                                update_MPC,
                                update_TTP)
            
            data_list = num_patients

        if(update_TTP):

            [num_theo_patients_per_placebo_arm,
             num_theo_patients_per_drug_arm,
             average_placebo_TTP,
             average_drug_TTP] = \
                 smart_algorithm(monthly_mean_min,
                                 monthly_mean_max,
                                 monthly_std_dev_min,
                                 monthly_std_dev_max,
                                 num_baseline_months,
                                 num_testing_months,
                                 minimum_required_baseline_seizure_count,
                                 placebo_mu,
                                 placebo_sigma,
                                 drug_mu,
                                 drug_sigma,
                                 num_trials_per_stat_power_estim,
                                 target_stat_power,
                                 num_extra_patients,
                                 stat_power_model,
                                 update_RR50,
                                 update_MPC,
                                 update_TTP)

            data_list = [num_theo_patients_per_placebo_arm, 
                         num_theo_patients_per_drug_arm,
                         average_placebo_TTP,
                         average_drug_TTP]

    else:

        raise ValueError('The smart_or_dumb parameter in \'empirical_guided_cost.py\' needs to either be \'smart\' or \'dumb\'')

    save_results(endpoint_name,
                 smart_or_dumb,
                 iter_index,
                 data_list)

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#

