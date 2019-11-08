import numpy as np
import keras.models as models
import matplotlib.pyplot as plt
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
                                  num_samples):

    num_monthly_std_devs = monthly_std_dev_max - monthly_std_dev_min + 1
    num_monthly_means    = monthly_mean_max    - monthly_mean_min    + 1
    keras_formatted_NV_model_placebo_arm_pop_hists = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_NV_model_drug_arm_pop_hists    = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))

    for sample_index in range(num_samples):

        NV_model_placebo_arm_patient_pop_params = \
            generate_NV_model_patient_pop_params(num_theo_patients_per_trial_arm,
                                                 one_or_two)
    
        NV_model_placebo_arm_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  NV_model_placebo_arm_patient_pop_params)
        
        keras_formatted_NV_model_placebo_arm_pop_hists[sample_index, :, :, 0] = NV_model_placebo_arm_pop_hist

        NV_model_drug_arm_patient_pop_params = \
            generate_NV_model_patient_pop_params(num_theo_patients_per_trial_arm,
                                                 one_or_two)
    
        NV_model_drug_arm_pop_hist = \
            convert_theo_pop_hist(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  NV_model_drug_arm_patient_pop_params)
        
        keras_formatted_NV_model_drug_arm_pop_hists[sample_index, :, :, 0] = NV_model_drug_arm_pop_hist
    
    return [keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists]


def predict_expected_stat_power_for_NV_model_endpoint(one_or_two,
                                                      endpoint_name,
                                                      num_theo_patients_per_trial_arm,
                                                      generic_stat_power_model_file_name,
                                                      monthly_mean_min,
                                                      monthly_mean_max,
                                                      monthly_std_dev_min,
                                                      monthly_std_dev_max,
                                                      num_samples):

    [keras_formatted_NV_model_placebo_arm_pop_hists, 
     keras_formatted_NV_model_drug_arm_pop_hists] = \
         generate_keras_formatted_data(num_theo_patients_per_trial_arm,
                                       one_or_two,
                                       monthly_mean_min,
                                       monthly_mean_max,
                                       monthly_std_dev_min,
                                       monthly_std_dev_max,
                                       num_samples)

    stat_power_model = models.load_model(endpoint_name + '_' + generic_stat_power_model_file_name + '.h5')
    nn_stat_powers = stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_nn_stat_power = np.mean(nn_stat_powers)

    return expected_nn_stat_power


def predict_NV_model_endpoint_stat_powers(num_theo_patients_per_trial_arm,
                                          generic_stat_power_model_file_name,
                                          monthly_mean_min,
                                          monthly_mean_max,
                                          monthly_std_dev_min,
                                          monthly_std_dev_max,
                                          num_samples):
    
    expected_NV_model_one_RR50_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('one',
                                                          'RR50',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)
    
    expected_NV_model_two_RR50_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('two',
                                                          'RR50',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)
    
    expected_NV_model_one_MPC_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('one',
                                                          'MPC',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)
    
    expected_NV_model_two_MPC_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('two',
                                                          'MPC',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)

    expected_NV_model_one_TTP_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('one',
                                                          'TTP',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)
    
    expected_NV_model_two_TTP_nn_stat_power = \
        predict_expected_stat_power_for_NV_model_endpoint('two',
                                                          'TTP',
                                                          num_theo_patients_per_trial_arm,
                                                          generic_stat_power_model_file_name,
                                                          monthly_mean_min,
                                                          monthly_mean_max,
                                                          monthly_std_dev_min,
                                                          monthly_std_dev_max,
                                                          num_samples)

    return [expected_NV_model_one_RR50_nn_stat_power,
            expected_NV_model_two_RR50_nn_stat_power,
            expected_NV_model_one_MPC_nn_stat_power,
            expected_NV_model_two_MPC_nn_stat_power,
            expected_NV_model_one_TTP_nn_stat_power,
            expected_NV_model_two_TTP_nn_stat_power]


def plot_predicted_statistical_powers(expected_NV_model_one_RR50_nn_stat_power,
                                      expected_NV_model_two_RR50_nn_stat_power,
                                      expected_NV_model_one_MPC_nn_stat_power,
                                      expected_NV_model_two_MPC_nn_stat_power,
                                      expected_NV_model_one_TTP_nn_stat_power,
                                      expected_NV_model_two_TTP_nn_stat_power):

    # data to plot
    n_groups = 3
    NV_model_one_endpoint_stat_powers = 100*np.array([expected_NV_model_one_RR50_nn_stat_power, 
                                                      expected_NV_model_one_MPC_nn_stat_power, 
                                                      expected_NV_model_one_TTP_nn_stat_power])

    NV_model_two_endpoint_stat_powers = 100*np.array([expected_NV_model_two_RR50_nn_stat_power, 
                                                      expected_NV_model_two_MPC_nn_stat_power, 
                                                      expected_NV_model_two_TTP_nn_stat_power])

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, NV_model_one_endpoint_stat_powers, bar_width,
    alpha=opacity,
    color='b',
    label='NV Model one')

    rects2 = plt.bar(index + bar_width, NV_model_two_endpoint_stat_powers, bar_width,
    alpha=opacity,
    color='g',
    label='NV Model two')

    plt.xlabel('endpoint')
    plt.ylabel('statistical power')
    plt.title('average predicted statistical powers')
    plt.xticks(index + bar_width/2, ('RR50', 'MPC', 'TTP'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('predicted NV model statistical powers.png')


if(__name__=='__main__'):

    num_theo_patients_per_trial_arm = 153
    generic_stat_power_model_file_name = 'stat_power_model'

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_samples = 100

    [expected_NV_model_one_RR50_nn_stat_power,
     expected_NV_model_two_RR50_nn_stat_power,
     expected_NV_model_one_MPC_nn_stat_power,
     expected_NV_model_two_MPC_nn_stat_power,
     expected_NV_model_one_TTP_nn_stat_power,
     expected_NV_model_two_TTP_nn_stat_power] = \
         predict_NV_model_endpoint_stat_powers(num_theo_patients_per_trial_arm,
                                               generic_stat_power_model_file_name,
                                               monthly_mean_min,
                                               monthly_mean_max,
                                               monthly_std_dev_min,
                                               monthly_std_dev_max,
                                               num_samples)

    plot_predicted_statistical_powers(expected_NV_model_one_RR50_nn_stat_power,
                                      expected_NV_model_two_RR50_nn_stat_power,
                                      expected_NV_model_one_MPC_nn_stat_power,
                                      expected_NV_model_two_MPC_nn_stat_power,
                                      expected_NV_model_one_TTP_nn_stat_power,
                                      expected_NV_model_two_TTP_nn_stat_power)

    print( '\nNV model one RR50 statistical power: ' + str(np.round(100*expected_NV_model_one_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model two RR50 statistical power: ' + str(np.round(100*expected_NV_model_two_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model one MPC  statistical power: ' + str(np.round(100*expected_NV_model_one_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model two MPC  statistical power: ' + str(np.round(100*expected_NV_model_two_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model one TTP  statistical power: ' + str(np.round(100*expected_NV_model_one_TTP_nn_stat_power,  3)) + ' %' + \
           '\nNV model two TTP  statistical power: ' + str(np.round(100*expected_NV_model_two_TTP_nn_stat_power,  3)) + ' %' + '\n' )
