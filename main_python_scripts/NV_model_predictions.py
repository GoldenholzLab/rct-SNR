import numpy as np
import keras.models as models
import json
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.patient_population_generation import generate_NV_model_patient_pop_params
from utility_code.patient_population_generation import convert_theo_pop_hist


def generate_keras_formatted_data(num_theo_patients_per_trial_arm,
                                  one_or_two,
                                  monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  num_trials):

    num_monthly_std_devs = monthly_std_dev_max - monthly_std_dev_min + 1
    num_monthly_means    = monthly_mean_max    - monthly_mean_min    + 1

    keras_formatted_NV_model_placebo_arm_pop_hists = np.zeros((num_trials, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_NV_model_drug_arm_pop_hists    = np.zeros((num_trials, num_monthly_std_devs, num_monthly_means, 1))

    NV_model_patient_pop_params = np.array([[], []]).reshape((0, 2))

    for trial_index in range(num_trials):

        NV_model_placebo_arm_patient_pop_params = \
            generate_NV_model_patient_pop_params(num_theo_patients_per_trial_arm,
                                                 one_or_two)
    
        NV_model_placebo_arm_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  NV_model_placebo_arm_patient_pop_params)
        
        keras_formatted_NV_model_placebo_arm_pop_hists[trial_index, :, :, 0] = NV_model_placebo_arm_pop_hist

        NV_model_drug_arm_patient_pop_params = \
            generate_NV_model_patient_pop_params(num_theo_patients_per_trial_arm,
                                                 one_or_two)

        NV_model_drug_arm_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  NV_model_drug_arm_patient_pop_params)
        
        keras_formatted_NV_model_drug_arm_pop_hists[trial_index, :, :, 0] = NV_model_drug_arm_pop_hist

        NV_model_patient_pop_params = np.concatenate((NV_model_patient_pop_params, NV_model_placebo_arm_patient_pop_params), 0)
        NV_model_patient_pop_params = np.concatenate((NV_model_patient_pop_params, NV_model_drug_arm_patient_pop_params),    0)

    return [keras_formatted_NV_model_placebo_arm_pop_hists, 
            keras_formatted_NV_model_drug_arm_pop_hists, 
            NV_model_patient_pop_params]


def predict_expected_stat_power_for_NV_model_endpoint(one_or_two,
                                                      num_theo_patients_per_trial_arm,
                                                      generic_stat_power_model_file_name,
                                                      monthly_mean_min,
                                                      monthly_mean_max,
                                                      monthly_std_dev_min,
                                                      monthly_std_dev_max,
                                                      num_trials):

    [keras_formatted_NV_model_placebo_arm_pop_hists, 
     keras_formatted_NV_model_drug_arm_pop_hists,
     NV_model_patient_pop_params] = \
         generate_keras_formatted_data(num_theo_patients_per_trial_arm,
                                       one_or_two,
                                       monthly_mean_min,
                                       monthly_mean_max,
                                       monthly_std_dev_min,
                                       monthly_std_dev_max,
                                       num_trials)
        
    RR50_stat_power_model = models.load_model('RR50_' + generic_stat_power_model_file_name + '.h5')
    RR50_nn_stat_powers = RR50_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_RR50_nn_stat_power = float(np.mean(RR50_nn_stat_powers))

    MPC_stat_power_model = models.load_model('MPC_' + generic_stat_power_model_file_name + '.h5')
    MPC_nn_stat_powers = MPC_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_MPC_nn_stat_power = float(np.mean(MPC_nn_stat_powers))

    TTP_stat_power_model = models.load_model('TTP_' + generic_stat_power_model_file_name + '.h5')
    TTP_nn_stat_powers = TTP_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_TTP_nn_stat_power = float(np.mean(TTP_nn_stat_powers))

    return [expected_NV_model_RR50_nn_stat_power, 
            expected_NV_model_MPC_nn_stat_power, 
            expected_NV_model_TTP_nn_stat_power, 
            NV_model_patient_pop_params]


def predict_NV_model_endpoint_stat_powers(num_theo_patients_per_trial_arm,
                                          generic_stat_power_model_file_name,
                                          monthly_mean_min,
                                          monthly_mean_max,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max,
                                          num_trials):
    
    [expected_NV_model_one_RR50_nn_stat_power, 
     expected_NV_model_one_MPC_nn_stat_power, 
     expected_NV_model_one_TTP_nn_stat_power, 
     NV_model_one_patient_pop_params] = \
         predict_expected_stat_power_for_NV_model_endpoint('one',
                                                           num_theo_patients_per_trial_arm,
                                                           generic_stat_power_model_file_name,
                                                           monthly_mean_min,
                                                           monthly_mean_max,
                                                           monthly_std_dev_min,
                                                           monthly_std_dev_max,
                                                           num_trials)
    
    [expected_NV_model_two_RR50_nn_stat_power, 
     expected_NV_model_two_MPC_nn_stat_power, 
     expected_NV_model_two_TTP_nn_stat_power, 
     NV_model_two_patient_pop_params] = \
         predict_expected_stat_power_for_NV_model_endpoint('two',
                                                           num_theo_patients_per_trial_arm,
                                                           generic_stat_power_model_file_name,
                                                           monthly_mean_min,
                                                           monthly_mean_max,
                                                           monthly_std_dev_min,
                                                           monthly_std_dev_max,
                                                           num_trials)
    
    NV_model_one_and_two_patient_pop_params = np.concatenate((NV_model_one_patient_pop_params, NV_model_two_patient_pop_params), 0)

    NV_model_one_and_two_patient_pop_hist = \
        convert_theo_pop_hist(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              NV_model_one_and_two_patient_pop_params)

    return [expected_NV_model_one_RR50_nn_stat_power, 
            expected_NV_model_one_MPC_nn_stat_power, 
            expected_NV_model_one_TTP_nn_stat_power,
            expected_NV_model_two_RR50_nn_stat_power, 
            expected_NV_model_two_MPC_nn_stat_power, 
            expected_NV_model_two_TTP_nn_stat_power,
            NV_model_one_and_two_patient_pop_hist]


def take_inputs_from_command_shell():

    '''
    num_theo_patients_per_trial_arm = 153
    generic_stat_power_model_file_name = 'stat_power_model'

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_trials = 100

    expected_NV_model_responses_file_name = 'expected_NV_model_responses'
    NV_model_hist_file_name = 'NV_model_histogram'
    '''

    num_theo_patients_per_trial_arm = int(sys.argv[1])
    generic_stat_power_model_file_name = sys.argv[2]

    monthly_mean_min    = int(sys.argv[3])
    monthly_mean_max    = int(sys.argv[4])
    monthly_std_dev_min = int(sys.argv[5])
    monthly_std_dev_max = int(sys.argv[6])

    num_trials = int(sys.argv[7])

    expected_NV_model_responses_file_name = sys.argv[8]
    NV_model_hist_file_name = sys.argv[9]

    return [num_theo_patients_per_trial_arm,
            generic_stat_power_model_file_name,
            monthly_mean_min,
            monthly_mean_max,
            monthly_std_dev_min,
            monthly_std_dev_max,
            num_trials,
            expected_NV_model_responses_file_name,
            NV_model_hist_file_name]


def save_responses_and_hist(expected_NV_model_one_RR50_nn_stat_power, 
                            expected_NV_model_one_MPC_nn_stat_power, 
                            expected_NV_model_one_TTP_nn_stat_power,
                            expected_NV_model_two_RR50_nn_stat_power, 
                            expected_NV_model_two_MPC_nn_stat_power, 
                            expected_NV_model_two_TTP_nn_stat_power,
                            expected_NV_model_responses_file_name,
                            NV_model_one_and_two_patient_pop_hist,
                            NV_model_hist_file_name):

    expected_NV_model_responses = \
        [expected_NV_model_one_RR50_nn_stat_power, 
         expected_NV_model_one_MPC_nn_stat_power, 
         expected_NV_model_one_TTP_nn_stat_power,
         expected_NV_model_two_RR50_nn_stat_power, 
         expected_NV_model_two_MPC_nn_stat_power, 
         expected_NV_model_two_TTP_nn_stat_power]

    with open(expected_NV_model_responses_file_name + '.json', 'w+') as json_file:

        json.dump(expected_NV_model_responses, json_file)
    
    with open(NV_model_hist_file_name + '.json', 'w+') as json_file:

        json.dump(NV_model_one_and_two_patient_pop_hist.tolist(), json_file)


if(__name__=='__main__'):

    [num_theo_patients_per_trial_arm,
     generic_stat_power_model_file_name,
     monthly_mean_min,
     monthly_mean_max,
     monthly_std_dev_min,
     monthly_std_dev_max,
     num_trials,
     expected_NV_model_responses_file_name,
     NV_model_hist_file_name] = \
         take_inputs_from_command_shell()
    
    [expected_NV_model_one_RR50_nn_stat_power, 
     expected_NV_model_one_MPC_nn_stat_power, 
     expected_NV_model_one_TTP_nn_stat_power,
     expected_NV_model_two_RR50_nn_stat_power, 
     expected_NV_model_two_MPC_nn_stat_power, 
     expected_NV_model_two_TTP_nn_stat_power,
     NV_model_one_and_two_patient_pop_hist] = \
         predict_NV_model_endpoint_stat_powers(num_theo_patients_per_trial_arm,
                                               generic_stat_power_model_file_name,
                                               monthly_mean_min,
                                               monthly_mean_max,
                                               monthly_std_dev_min,
                                               monthly_std_dev_max,
                                               num_trials)
    
    save_responses_and_hist(expected_NV_model_one_RR50_nn_stat_power, 
                            expected_NV_model_one_MPC_nn_stat_power, 
                            expected_NV_model_one_TTP_nn_stat_power,
                            expected_NV_model_two_RR50_nn_stat_power, 
                            expected_NV_model_two_MPC_nn_stat_power, 
                            expected_NV_model_two_TTP_nn_stat_power,
                            expected_NV_model_responses_file_name,
                            NV_model_one_and_two_patient_pop_hist,
                            NV_model_hist_file_name)

