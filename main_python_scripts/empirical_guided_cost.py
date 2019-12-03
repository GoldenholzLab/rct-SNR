import numpy as np
import json
import keras.models as models
import copy
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary_with_minimum_count
from utility_code.patient_population_generation import convert_theo_pop_hist
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
                                                      28,
                                                      minimum_required_baseline_seizure_count)
    
        monthly_mean_hat    = np.int_(np.round(28*np.mean(baseline_monthly_seizure_diary)))
        monthly_std_dev_hat = np.int_(np.round(np.sqrt(28)*np.std(baseline_monthly_seizure_diary)))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )

        estimated_location_overdispersed = monthly_std_dev_hat > np.sqrt(monthly_mean_hat)

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and estimated_location_overdispersed

    return [monthly_mean_hat, monthly_std_dev_hat]


def calculate_SNRs(theo_placebo_arm_patient_pop_params_hat_list,
                   theo_drug_arm_patient_pop_params_hat_list,
                   monthly_mean_hat,
                   monthly_std_dev_hat,
                   current_trial_arm,
                   num_extra_patients,
                   monthly_mean_min,
                   monthly_mean_max,
                   monthly_std_dev_min,
                   monthly_std_dev_max,
                   RR50_stat_power_model,
                   MPC_stat_power_model,
                   TTP_stat_power_model):

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
        
    RR50_current_power = \
        np.squeeze(RR50_stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat, 
                                                  keras_formatted_theo_drug_arm_patient_pop_params_hat]))
        
    RR50_power_with_extra_patients = \
        np.squeeze(RR50_stat_power_model.predict([keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat, 
                                                  keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat]))
        
    RR50_loc_SNR = RR50_power_with_extra_patients - RR50_current_power

    postive_RR50_SNR = RR50_loc_SNR > 0
    
    MPC_current_power = \
        np.squeeze(MPC_stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat, 
                                                 keras_formatted_theo_drug_arm_patient_pop_params_hat]))
        
    MPC_power_with_extra_patients = \
        np.squeeze(MPC_stat_power_model.predict([keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat, 
                                                 keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat]))
        
    MPC_loc_SNR = MPC_power_with_extra_patients - MPC_current_power

    postive_MPC_SNR = MPC_loc_SNR > 0

    TTP_current_power = \
        np.squeeze(TTP_stat_power_model.predict([keras_formatted_theo_placebo_arm_patient_pop_params_hat, 
                                                 keras_formatted_theo_drug_arm_patient_pop_params_hat]))
        
    TTP_power_with_extra_patients = \
        np.squeeze(TTP_stat_power_model.predict([keras_formatted_tmp_theo_placebo_arm_patient_pop_params_hat, 
                                                 keras_formatted_tmp_theo_drug_arm_patient_pop_params_hat]))
        
    TTP_loc_SNR = TTP_power_with_extra_patients - TTP_current_power

    postive_TTP_SNR = TTP_loc_SNR > 0

    '''
    print( np.array([[RR50_current_power, RR50_power_with_extra_patients], 
                     [MPC_current_power,  MPC_power_with_extra_patients], 
                     [TTP_current_power,  TTP_power_with_extra_patients]]) )
    '''
    #print(np.round(100*np.array([RR50_loc_SNR, MPC_loc_SNR, TTP_loc_SNR]), 3))

    return [postive_RR50_SNR, postive_MPC_SNR, postive_TTP_SNR]


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
                   target_stat_power):

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
    

    update_RR50 = True
    update_MPC  = True
    update_TTP  = True
    underpowered_endpoints = True

    num_RR50_patients = num_patients
    num_MPC_patients  = num_patients
    num_placebo_TTP_patients  = num_theo_patients_per_placebo_arm
    num_drug_TTP_patients     = num_theo_patients_per_drug_arm
    final_average_placebo_TTP = 0
    final_average_drug_TTP    = 0

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
            num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1
            current_trial_arm = 'drug' 

        elif( current_trial_arm == 'drug' ):
            theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1
            current_trial_arm = 'placebo'

        num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

        [RR50_stat_power, 
         MPC_stat_power, 
         TTP_stat_power, 
         average_placebo_TTP, 
         average_drug_TTP] = \
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
        
        underpowered_endpoints = update_RR50 or update_MPC or update_TTP

        if( update_RR50 ):
            if( RR50_stat_power < target_stat_power ):
                num_RR50_patients = num_patients
            else:
                update_RR50 = False
        if( update_MPC ):
            if( MPC_stat_power  < target_stat_power ):
                num_MPC_patients = num_patients
            else:
                update_MPC = False
        if( update_TTP ):
            if( TTP_stat_power  < target_stat_power ):
                num_placebo_TTP_patients  = num_theo_patients_per_placebo_arm
                num_drug_TTP_patients     = num_theo_patients_per_drug_arm
            else:
                update_TTP = False
                final_average_placebo_TTP = average_placebo_TTP
                final_average_drug_TTP    = average_drug_TTP
        
        print( '\n' + str( 100*np.array([RR50_stat_power,   MPC_stat_power,   TTP_stat_power]) ) + '\n' + \
                      str( [num_RR50_patients, num_MPC_patients, num_placebo_TTP_patients + num_drug_TTP_patients] ) + '\n' )

    return [num_RR50_patients, 
            num_MPC_patients, 
            num_placebo_TTP_patients,
            num_drug_TTP_patients,
            final_average_placebo_TTP, 
            final_average_drug_TTP]


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
                    generic_stat_power_model_file_name):

    RR50_stat_power_model = models.load_model('RR50_' + generic_stat_power_model_file_name + '.h5')
    MPC_stat_power_model  = models.load_model('MPC_'  + generic_stat_power_model_file_name + '.h5')
    TTP_stat_power_model  = models.load_model('TTP_'  + generic_stat_power_model_file_name + '.h5')

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

    update_RR50 = True
    update_MPC  = True
    update_TTP  = True
    underpowered_endpoints = True

    num_RR50_patients = num_patients
    num_MPC_patients  = num_patients
    num_placebo_TTP_patients  = num_theo_patients_per_placebo_arm
    num_drug_TTP_patients     = num_theo_patients_per_drug_arm
    final_average_placebo_TTP = 0
    final_average_drug_TTP    = 0

    while(underpowered_endpoints):

        [monthly_mean_hat, 
         monthly_std_dev_hat] = \
             estimate_patient_loc(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  num_baseline_months,
                                  minimum_required_baseline_seizure_count)

        [postive_RR50_SNR, 
         postive_MPC_SNR, 
         postive_TTP_SNR] = \
             calculate_SNRs(theo_placebo_arm_patient_pop_params_hat_list,
                            theo_drug_arm_patient_pop_params_hat_list,
                            monthly_mean_hat,
                            monthly_std_dev_hat,
                            current_trial_arm,
                            num_extra_patients,
                            monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max,
                            RR50_stat_power_model,
                            MPC_stat_power_model,
                            TTP_stat_power_model)
        
        acceptable_RR50_SNR = (not update_RR50) or (update_RR50 and postive_RR50_SNR)
        acceptable_MPC_SNR  = (not update_MPC)  or (update_MPC  and postive_MPC_SNR)
        acceptable_TTP_SNR  = (not update_TTP)  or (update_TTP  and postive_TTP_SNR)
        acceptable_SNR = acceptable_RR50_SNR and acceptable_MPC_SNR and acceptable_TTP_SNR

        if(acceptable_SNR):

            if( current_trial_arm == 'placebo' ):
                theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1
                current_trial_arm = 'drug'

            elif( current_trial_arm == 'drug' ):
                theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
                num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1
                current_trial_arm = 'placebo'

            num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

            [RR50_stat_power, 
             MPC_stat_power, 
             TTP_stat_power, 
             average_placebo_TTP, 
             average_drug_TTP]  = \
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
        
            underpowered_endpoints = update_RR50 or update_MPC or update_TTP

            if( update_RR50 ):
                if( RR50_stat_power < target_stat_power ):
                    num_RR50_patients = num_patients
                else:
                    update_RR50 = False
            if( update_MPC ):
                if( MPC_stat_power  < target_stat_power ):
                    num_MPC_patients = num_patients
                else:
                    update_MPC = False
            if( update_TTP ):
                if( TTP_stat_power  < target_stat_power ):
                    num_placebo_TTP_patients  = num_theo_patients_per_placebo_arm
                    num_drug_TTP_patients     = num_theo_patients_per_drug_arm
                else:
                    update_TTP = False
                    final_average_placebo_TTP = average_placebo_TTP
                    final_average_drug_TTP    = average_drug_TTP

        
            print('\n' + 'accepted: ' + str([monthly_mean_hat, monthly_std_dev_hat]) + \
                  '\n' +                str( 100*np.array([RR50_stat_power, MPC_stat_power, TTP_stat_power]) ) + '\n' + \
                                        str( [num_RR50_patients, num_MPC_patients, num_placebo_TTP_patients + num_drug_TTP_patients] )  + '\n' )
        
        else:

            print('rejected: ' + str([monthly_mean_hat, monthly_std_dev_hat]))

    return [num_RR50_patients, 
            num_MPC_patients, 
            num_placebo_TTP_patients,
            num_drug_TTP_patients,
            final_average_placebo_TTP, 
            final_average_drug_TTP]


def save_results(smart_or_dumb,
                 iter_index,
                 num_RR50_patients, 
                 num_MPC_patients, 
                 num_placebo_TTP_patients,
                 num_drug_TTP_patients, 
                 final_average_placebo_TTP, 
                 final_average_drug_TTP):

    folder_name = os.getcwd() + '/' + smart_or_dumb +'_data'

    if( not os.path.isdir(folder_name) ):

        os.makedirs(folder_name)
    
    with open(folder_name + '/' + str(iter_index) + '.json', 'w+') as json_file:
        
        json.dump([num_RR50_patients, 
                   num_MPC_patients, 
                   num_placebo_TTP_patients,
                   num_drug_TTP_patients, 
                   final_average_placebo_TTP, 
                   final_average_drug_TTP   ], json_file)


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
    iter_index    = int(sys.argv[17])

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
     iter_index] = \
         take_inputs_from_command_shell()

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#

    print(smart_or_dumb + ', iter #' + str(iter_index))

    if(smart_or_dumb == 'dumb'):
    
        [num_RR50_patients, 
         num_MPC_patients, 
         num_placebo_TTP_patients,
         num_drug_TTP_patients,
         final_average_placebo_TTP, 
         final_average_drug_TTP] = \
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
                            target_stat_power)  

    elif(smart_or_dumb == 'smart'):

        [num_RR50_patients, 
         num_MPC_patients, 
         num_placebo_TTP_patients,
         num_drug_TTP_patients,
         final_average_placebo_TTP, 
         final_average_drug_TTP] = \
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
                             generic_stat_power_model_file_name)

    else:

        raise ValueError('The smart_or_dumb parameter in \'empirical_guided_cost.py\' needs to either be \'smart\' or \'dumb\'')

    print(smart_or_dumb + ' : ' + str([num_RR50_patients, 
                                       num_MPC_patients, 
                                       num_placebo_TTP_patients,
                                       num_drug_TTP_patients, 
                                       final_average_placebo_TTP, 
                                       final_average_drug_TTP]))

    save_results(smart_or_dumb,
                 iter_index,
                 num_RR50_patients, 
                 num_MPC_patients, 
                 num_placebo_TTP_patients,
                 num_drug_TTP_patients, 
                 final_average_placebo_TTP, 
                 final_average_drug_TTP)

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#

