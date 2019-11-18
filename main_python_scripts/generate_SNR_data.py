import numpy as np
import sys
import os
import keras.models as models
import json
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import convert_theo_pop_hist


def generate_hists_with_and_without_loc(monthly_mean_min,
                                        monthly_mean_max,
                                        monthly_std_dev_min,
                                        monthly_std_dev_max,
                                        current_monthly_mean,
                                        current_monthly_std_dev,
                                        num_theo_patients_per_trial_arm_in_snr_map,
                                        num_theo_patients_per_trial_arm_in_snr_map_loc,
                                        loc_in_placebo_or_drug):

    if(loc_in_placebo_or_drug != 'placebo' and loc_in_placebo_or_drug != 'drug'):

        raise ValueError("The \'loc_in_placebo_or_drug\' parameter must either be \'placebo\' or \'drug\'")

    theo_placebo_arm_patient_pop_params_wo_loc = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm_in_snr_map)
    
    theo_drug_arm_patient_pop_params_wo_loc = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm_in_snr_map)

    theo_placebo_arm_patient_pop_params_with_loc = np.copy(theo_placebo_arm_patient_pop_params_wo_loc)
    theo_drug_arm_patient_pop_params_with_loc    = np.copy(theo_drug_arm_patient_pop_params_wo_loc)

    patient_pop_params_per_one_loc = np.transpose(np.vstack([np.full(num_theo_patients_per_trial_arm_in_snr_map_loc, current_monthly_mean),
                                                             np.full(num_theo_patients_per_trial_arm_in_snr_map_loc, current_monthly_std_dev)]))

    if(loc_in_placebo_or_drug == 'placebo'):
        theo_placebo_arm_patient_pop_params_with_loc = np.vstack([theo_placebo_arm_patient_pop_params_with_loc, patient_pop_params_per_one_loc])
    elif(loc_in_placebo_or_drug == 'drug'):
        theo_drug_arm_patient_pop_params_with_loc    = np.vstack([theo_drug_arm_patient_pop_params_with_loc,    patient_pop_params_per_one_loc])

    theo_placebo_arm_trial_arm_pop_wo_loc_hist = \
        convert_theo_pop_hist(1,
                              16,
                              1,
                              16,
                              theo_placebo_arm_patient_pop_params_wo_loc)
    
    theo_drug_arm_trial_arm_pop_wo_loc_hist = \
        convert_theo_pop_hist(1,
                              16,
                              1,
                              16,
                              theo_drug_arm_patient_pop_params_wo_loc)

    theo_placebo_arm_trial_arm_pop_with_loc_hist = \
        convert_theo_pop_hist(1,
                              16,
                              1,
                              16,
                              theo_placebo_arm_patient_pop_params_with_loc)
    
    theo_drug_arm_trial_arm_pop_with_loc_hist = \
        convert_theo_pop_hist(1,
                              16,
                              1,
                              16,
                              theo_drug_arm_patient_pop_params_with_loc)
    
    return [theo_placebo_arm_trial_arm_pop_wo_loc_hist,
            theo_drug_arm_trial_arm_pop_wo_loc_hist,
            theo_placebo_arm_trial_arm_pop_with_loc_hist,
            theo_drug_arm_trial_arm_pop_with_loc_hist]


def generate_keras_formatted_hists_with_and_without_loc(monthly_mean_min,
                                                        monthly_mean_max,
                                                        monthly_std_dev_min,
                                                        monthly_std_dev_max,
                                                        num_hists_per_trial_arm,
                                                        current_monthly_mean,
                                                        current_monthly_std_dev,
                                                        num_theo_patients_per_trial_arm_in_snr_map,
                                                        num_theo_patients_per_trial_arm_in_snr_map_loc,
                                                        loc_in_placebo_or_drug):
    
    keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists   = np.zeros((num_hists_per_trial_arm, 16, 16, 1))
    keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists      = np.zeros((num_hists_per_trial_arm, 16, 16, 1))
    keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists = np.zeros((num_hists_per_trial_arm, 16, 16, 1))
    keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists    = np.zeros((num_hists_per_trial_arm, 16, 16, 1))

    for hist_index in range(num_hists_per_trial_arm):

        [tmp_theo_placebo_arm_trial_arm_pop_wo_loc_hist,
         tmp_theo_drug_arm_trial_arm_pop_wo_loc_hist,
         tmp_theo_placebo_arm_trial_arm_pop_with_loc_hist,
         tmp_theo_drug_arm_trial_arm_pop_with_loc_hist] = \
             generate_hists_with_and_without_loc(monthly_mean_min,
                                                 monthly_mean_max,
                                                 monthly_std_dev_min,
                                                 monthly_std_dev_max,
                                                 current_monthly_mean,
                                                 current_monthly_std_dev,
                                                 num_theo_patients_per_trial_arm_in_snr_map,
                                                 num_theo_patients_per_trial_arm_in_snr_map_loc,
                                                 loc_in_placebo_or_drug)

        keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists[hist_index, :, :, 0]   = tmp_theo_placebo_arm_trial_arm_pop_wo_loc_hist
        keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists[hist_index, :, :, 0]      = tmp_theo_drug_arm_trial_arm_pop_wo_loc_hist
        keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists[hist_index, :, :, 0] = tmp_theo_placebo_arm_trial_arm_pop_with_loc_hist
        keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists[hist_index, :, :, 0]    = tmp_theo_drug_arm_trial_arm_pop_with_loc_hist

    return [keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists,
            keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists,
            keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists,
            keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists]


def generate_keras_formatted_hists(monthly_mean_min,
                                   monthly_mean_max,
                                   monthly_std_dev_min,
                                   monthly_std_dev_max,
                                   num_hists_per_trial_arm,
                                   current_monthly_mean,
                                   current_monthly_std_dev,
                                   num_theo_patients_per_trial_arm_in_snr_map,
                                   num_theo_patients_per_trial_arm_in_snr_map_loc):

    [keras_formatted_theo_placebo_arm_trial_arm_pop_wo_placebo_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_wo_placebo_loc_hists,
     keras_formatted_theo_placebo_arm_trial_arm_pop_with_placebo_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_with_placebo_loc_hists] = \
         generate_keras_formatted_hists_with_and_without_loc(monthly_mean_min,
                                                             monthly_mean_max,
                                                             monthly_std_dev_min,
                                                             monthly_std_dev_max,
                                                             num_hists_per_trial_arm,
                                                             current_monthly_mean,
                                                             current_monthly_std_dev,
                                                             num_theo_patients_per_trial_arm_in_snr_map,
                                                             num_theo_patients_per_trial_arm_in_snr_map_loc,
                                                             'placebo')
    
    [keras_formatted_theo_placebo_arm_trial_arm_pop_wo_drug_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_wo_drug_loc_hists,
     keras_formatted_theo_placebo_arm_trial_arm_pop_with_drug_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_with_drug_loc_hists] = \
         generate_keras_formatted_hists_with_and_without_loc(monthly_mean_min,
                                                             monthly_mean_max,
                                                             monthly_std_dev_min,
                                                             monthly_std_dev_max,
                                                             num_hists_per_trial_arm,
                                                             current_monthly_mean,
                                                             current_monthly_std_dev,
                                                             num_theo_patients_per_trial_arm_in_snr_map,
                                                             num_theo_patients_per_trial_arm_in_snr_map_loc,
                                                             'drug')
    
    keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists = \
        np.concatenate([keras_formatted_theo_placebo_arm_trial_arm_pop_wo_placebo_loc_hists, 
                        keras_formatted_theo_placebo_arm_trial_arm_pop_wo_drug_loc_hists],   0)

    keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists = \
        np.concatenate([keras_formatted_theo_drug_arm_trial_arm_pop_wo_placebo_loc_hists, 
                        keras_formatted_theo_drug_arm_trial_arm_pop_wo_drug_loc_hists],   0)
    
    keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists = \
        np.concatenate([keras_formatted_theo_placebo_arm_trial_arm_pop_with_placebo_loc_hists, 
                        keras_formatted_theo_placebo_arm_trial_arm_pop_with_drug_loc_hists],   0)

    keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists = \
        np.concatenate([keras_formatted_theo_drug_arm_trial_arm_pop_with_placebo_loc_hists, 
                        keras_formatted_theo_drug_arm_trial_arm_pop_with_drug_loc_hists],   0)

    return [keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists,
            keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists,
            keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists,
            keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists]


def generate_SNRs_for_loc(monthly_mean_min,
                          monthly_mean_max,
                          monthly_std_dev_min,
                          monthly_std_dev_max,
                          num_hists_per_trial_arm,
                          current_monthly_mean,
                          current_monthly_std_dev,
                          num_theo_patients_per_trial_arm_in_snr_map,
                          num_theo_patients_per_trial_arm_in_snr_map_loc,
                          stat_power_model):

    [keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists,
     keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists,
     keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists] = \
         generate_keras_formatted_hists(monthly_mean_min,
                                        monthly_mean_max,
                                        monthly_std_dev_min,
                                        monthly_std_dev_max,
                                        num_hists_per_trial_arm,
                                        current_monthly_mean,
                                        current_monthly_std_dev,
                                        num_theo_patients_per_trial_arm_in_snr_map,
                                        num_theo_patients_per_trial_arm_in_snr_map_loc)

    stat_power_predictions_wo_loc = \
        np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists, 
                                             keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists]))
    stat_power_predictions_with_loc = \
        np.squeeze(stat_power_model.predict([keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists, 
                                             keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists]))

    stat_power_wo_loc   = 100*np.mean(stat_power_predictions_wo_loc)
    stat_power_with_loc = 100*np.mean(stat_power_predictions_with_loc)
    SNR_at_loc = 100*np.mean(stat_power_predictions_with_loc - stat_power_predictions_wo_loc)

    return [SNR_at_loc, stat_power_with_loc, stat_power_wo_loc]


def generate_SNR_map(monthly_mean_min,
                     monthly_mean_max,
                     monthly_std_dev_min,
                     monthly_std_dev_max,
                     num_hists_per_trial_arm,
                     num_theo_patients_per_trial_arm_in_snr_map,
                     num_theo_patients_per_trial_arm_in_snr_map_loc,
                     endpoint_name):

    '''

    Have to remember to investigate some parameters in here

    '''

    stat_power_model = models.load_model(endpoint_name + '_stat_power_model.h5')

    SNR_map = np.zeros((15, 15))

    for monthly_mean_index in range(15):
        for monthly_std_dev_index in range(15):

            monthly_std_dev =  monthly_std_dev_index + 1
            monthly_mean    = monthly_mean_index + 1

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                [SNR_at_loc, 
                 stat_power_with_loc, 
                 stat_power_wo_loc] = \
                     generate_SNRs_for_loc(monthly_mean_min,
                                           monthly_mean_max,
                                           monthly_std_dev_min,
                                           monthly_std_dev_max,
                                           num_hists_per_trial_arm,
                                           monthly_mean,
                                           monthly_std_dev,
                                           num_theo_patients_per_trial_arm_in_snr_map,
                                           num_theo_patients_per_trial_arm_in_snr_map_loc,
                                           stat_power_model)
                
                print('[' + str(monthly_mean) + ', ' + str(monthly_std_dev) + ']: ' + str(np.round(SNR_at_loc, 3)) + \
                      ', ' + str(np.round(stat_power_with_loc, 3))  + ', ' + str(np.round(stat_power_wo_loc, 3)))    
                

                SNR_map[15 - monthly_std_dev_index - 1, monthly_mean_index] = SNR_at_loc
    
    return SNR_map


def take_inputs_from_command_shell():

    num_theo_patients_per_trial_arm_in_snr_map     = int(sys.argv[1])
    num_theo_patients_per_trial_arm_in_snr_map_loc = int(sys.argv[2])

    num_hists_per_trial_arm = int(sys.argv[3])

    endpoint_name = sys.argv[4]

    return [num_theo_patients_per_trial_arm_in_snr_map,
            num_theo_patients_per_trial_arm_in_snr_map_loc,
            num_hists_per_trial_arm,
            endpoint_name]


if (__name__=="__main__"):

    monthly_mean_min    = 2
    monthly_mean_max    = 15
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    [num_theo_patients_per_trial_arm_in_snr_map,
     num_theo_patients_per_trial_arm_in_snr_map_loc,
     num_hists_per_trial_arm,
     endpoint_name] = \
         take_inputs_from_command_shell()

    SNR_map = \
        generate_SNR_map(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_hists_per_trial_arm,
                         num_theo_patients_per_trial_arm_in_snr_map,
                         num_theo_patients_per_trial_arm_in_snr_map_loc,
                         endpoint_name)
    
    
    with open(endpoint_name + '_SNR_data.json', 'w+') as json_file:
        json.dump(SNR_map.tolist(), json_file)

