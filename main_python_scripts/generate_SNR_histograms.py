import numpy as np
import sys
import os
import keras.models as models
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import generate_theo_patient_pop_params
from utility_code.patient_population_generation import convert_theo_pop_hist


def generate_hists_with_and_without_loc(monthly_mean_min,
                                        monthly_mean_max,
                                        monthly_std_dev_min,
                                        monthly_std_dev_max,
                                        current_monthly_mean,
                                        current_monthly_std_dev,
                                        num_theo_patients_per_trial_arm,
                                        num_theo_patients_per_trial_arm_in_loc,
                                        loc_in_placebo_or_drug):

    if(loc_in_placebo_or_drug != 'placebo' and loc_in_placebo_or_drug != 'drug'):

        raise ValueError("The \'loc_in_placebo_or_drug\' parameter must either be \'placebo\' or \'drug\'")

    theo_placebo_arm_patient_pop_params_wo_loc = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    theo_drug_arm_patient_pop_params_wo_loc = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)

    theo_placebo_arm_patient_pop_params_with_loc = theo_placebo_arm_patient_pop_params_wo_loc
    theo_drug_arm_patient_pop_params_with_loc    = theo_drug_arm_patient_pop_params_wo_loc

    patient_pop_params_per_one_loc = np.transpose(np.vstack([np.full(num_theo_patients_per_trial_arm_in_loc, current_monthly_mean),
                                                             np.full(num_theo_patients_per_trial_arm_in_loc, current_monthly_std_dev)]))

    if(loc_in_placebo_or_drug == 'placebo'):
        theo_placebo_arm_patient_pop_params_with_loc = np.vstack([theo_placebo_arm_patient_pop_params_with_loc, patient_pop_params_per_one_loc])
    elif(loc_in_placebo_or_drug == 'drug'):
        theo_drug_arm_patient_pop_params_with_loc = np.vstack([theo_drug_arm_patient_pop_params_with_loc, patient_pop_params_per_one_loc])

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
                              theo_drug_arm_patient_pop_params_wo_loc)
    
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
                                                        num_theo_patients_per_trial_arm,
                                                        num_theo_patients_per_trial_arm_in_loc,
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
                                                 num_theo_patients_per_trial_arm,
                                                 num_theo_patients_per_trial_arm_in_loc,
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
                                   num_theo_patients_per_trial_arm,
                                   num_theo_patients_per_trial_arm_in_loc):

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
                                                             num_theo_patients_per_trial_arm,
                                                             num_theo_patients_per_trial_arm_in_loc,
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
                                                             num_theo_patients_per_trial_arm,
                                                             num_theo_patients_per_trial_arm_in_loc,
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
                          num_theo_patients_per_trial_arm,
                          num_theo_patients_per_trial_arm_in_loc,
                          RR50_stat_power_model):

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
                                        num_theo_patients_per_trial_arm,
                                        num_theo_patients_per_trial_arm_in_loc)
    
    RR50_stat_power_predictions_wo_loc = \
        np.squeeze(RR50_stat_power_model.predict([keras_formatted_theo_placebo_arm_trial_arm_pop_wo_loc_hists, 
                                              keras_formatted_theo_drug_arm_trial_arm_pop_wo_loc_hists]))
    RR50_stat_power_predictions_with_loc = \
        np.squeeze(RR50_stat_power_model.predict([keras_formatted_theo_placebo_arm_trial_arm_pop_with_loc_hists, 
                                                  keras_formatted_theo_drug_arm_trial_arm_pop_with_loc_hists]))

    RR50_SNR_at_loc = 100*np.mean(RR50_stat_power_predictions_with_loc - RR50_stat_power_predictions_wo_loc)

    return RR50_SNR_at_loc


if (__name__=="__main__"):

    monthly_mean_min    = 2
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_theo_patients_per_trial_arm = 50
    num_theo_patients_per_trial_arm_in_loc = 20

    num_hists_per_trial_arm = 100

    RR50_stat_power_model = models.load_model('RR50_stat_power_model.h5')

    for monthly_mean in range(1, 16 + 1):
        for monthly_std_dev in range(1, 16 + 1):

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                RR50_SNR_at_loc = \
                    generate_SNRs_for_loc(monthly_mean_min,
                                          monthly_mean_max,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max,
                                          num_hists_per_trial_arm,
                                          monthly_mean,
                                          monthly_std_dev,
                                          num_theo_patients_per_trial_arm,
                                          num_theo_patients_per_trial_arm_in_loc,
                                          RR50_stat_power_model)

                print('[' + str(monthly_mean) + ', ' + str(monthly_std_dev) + ']: ' + str(RR50_SNR_at_loc))    
    
