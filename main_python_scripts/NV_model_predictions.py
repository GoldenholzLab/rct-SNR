import numpy as np
import keras.models as models
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import string
from PIL import Image
import io
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

    NV_model_patient_pop_params = np.array([[], []]).reshape((0, 2))

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
                                                      num_samples):

    [keras_formatted_NV_model_placebo_arm_pop_hists, 
     keras_formatted_NV_model_drug_arm_pop_hists,
     NV_model_patient_pop_params] = \
         generate_keras_formatted_data(num_theo_patients_per_trial_arm,
                                       one_or_two,
                                       monthly_mean_min,
                                       monthly_mean_max,
                                       monthly_std_dev_min,
                                       monthly_std_dev_max,
                                       num_samples)
        
    RR50_stat_power_model = models.load_model('RR50_' + generic_stat_power_model_file_name + '.h5')
    RR50_nn_stat_powers = RR50_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_RR50_nn_stat_power = np.mean(RR50_nn_stat_powers)

    MPC_stat_power_model = models.load_model('MPC_' + generic_stat_power_model_file_name + '.h5')
    MPC_nn_stat_powers = MPC_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_MPC_nn_stat_power = np.mean(MPC_nn_stat_powers)

    TTP_stat_power_model = models.load_model('TTP_' + generic_stat_power_model_file_name + '.h5')
    TTP_nn_stat_powers = TTP_stat_power_model.predict([keras_formatted_NV_model_placebo_arm_pop_hists, keras_formatted_NV_model_drug_arm_pop_hists])
    expected_NV_model_TTP_nn_stat_power = np.mean(TTP_nn_stat_powers)

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
                                          num_samples):
    
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
                                                           num_samples)
    
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
                                                           num_samples)
    
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


def plot_predicted_statistical_powers_and_NV_model_hists(expected_NV_model_one_RR50_nn_stat_power,
                                                         expected_NV_model_two_RR50_nn_stat_power,
                                                         expected_NV_model_one_MPC_nn_stat_power,
                                                         expected_NV_model_two_MPC_nn_stat_power,
                                                         expected_NV_model_one_TTP_nn_stat_power,
                                                         expected_NV_model_two_TTP_nn_stat_power,
                                                         NV_model_one_and_two_patient_pop_hist,
                                                         monthly_mean_axis_start,
                                                         monthly_mean_axis_stop,
                                                         monthly_mean_axis_step,
                                                         monthly_mean_tick_spacing,
                                                         monthly_std_dev_axis_start,
                                                         monthly_std_dev_axis_stop,
                                                         monthly_std_dev_axis_step,
                                                         monthly_std_dev_tick_spacing):

    monthly_mean_tick_labels = np.arange(monthly_mean_axis_start, monthly_mean_axis_stop + monthly_mean_tick_spacing, monthly_mean_tick_spacing)
    monthly_std_dev_tick_labels = np.arange(monthly_std_dev_axis_start, monthly_std_dev_axis_stop + monthly_std_dev_tick_spacing, monthly_std_dev_tick_spacing)

    monthly_mean_ticks = monthly_mean_tick_labels/monthly_mean_axis_step + 0.5 - 1
    monthly_std_dev_ticks = monthly_std_dev_tick_labels/monthly_std_dev_axis_step + 0.5 - 1

    monthly_std_dev_ticks = np.flip(monthly_std_dev_ticks, 0)

    n_groups = 3
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    NV_model_one_endpoint_stat_powers = 100*np.array([expected_NV_model_one_RR50_nn_stat_power, 
                                                      expected_NV_model_one_MPC_nn_stat_power, 
                                                      expected_NV_model_one_TTP_nn_stat_power])

    NV_model_two_endpoint_stat_powers = 100*np.array([expected_NV_model_two_RR50_nn_stat_power, 
                                                      expected_NV_model_two_MPC_nn_stat_power, 
                                                      expected_NV_model_two_TTP_nn_stat_power])

    fig = plt.figure(figsize=(20, 6))

    ax = plt.subplot(1,2,1)

    ax = sns.heatmap(NV_model_one_and_two_patient_pop_hist, cmap='RdBu_r', cbar_kws={'label':'NUmber of patients'})
    ax.set_xticks(monthly_mean_ticks)
    ax.set_xticklabels(monthly_mean_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_std_dev_ticks)
    ax.set_yticklabels(monthly_std_dev_tick_labels, rotation='horizontal')
    ax.set_xlabel('monthly seizure count mean')
    ax.set_ylabel('monthly seizure count standard deviation')
    ax.title.set_text('2D histograms of patients from NV model 1 and 2')
    ax.text(-0.2, 1, string.ascii_uppercase[0] + ')', 
                transform=ax.transAxes, size=15, weight='bold')

    plt.subplot(1,2,2)

    plt.bar(index, NV_model_one_endpoint_stat_powers, bar_width,
    alpha=opacity,
    color='b',
    label='NV Model one')

    plt.bar(index + bar_width, NV_model_two_endpoint_stat_powers, bar_width,
    alpha=opacity,
    color='g',
    label='NV Model two')

    plt.xlabel('endpoint')
    plt.ylabel('statistical power')
    plt.title('average predicted statistical powers')
    plt.xticks(index + bar_width/2, ('RR50', 'MPC', 'TTP'))
    plt.yticks(np.arange(0, 110, 10))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.yaxis.grid(True)
    plt.legend()
    ax.text(-0.2, 1, string.ascii_uppercase[1] + ')', 
                transform=ax.transAxes, size=15, weight='bold')

    plt.subplots_adjust(wspace = .25)

    png1 = io.BytesIO()
    fig.savefig(png1, dpi = 600, bbox_inches = 'tight', format = 'png')
    png2 = Image.open(png1)
    png2.save('Romero-fig4.tiff')
    png1.close()


if(__name__=='__main__'):

    num_theo_patients_per_trial_arm = 153
    generic_stat_power_model_file_name = 'stat_power_model'

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_samples = 100

    monthly_mean_axis_start   = monthly_mean_min
    monthly_mean_axis_stop    = monthly_mean_max - 1
    monthly_mean_axis_step    = 1
    monthly_mean_tick_spacing = 1

    monthly_std_dev_axis_start   = monthly_std_dev_min
    monthly_std_dev_axis_stop    = monthly_std_dev_max - 1
    monthly_std_dev_axis_step    = 1
    monthly_std_dev_tick_spacing = 1
    
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
                                               num_samples)
    
    print( '\nNV model one RR50 statistical power: ' + str(np.round(100*expected_NV_model_one_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model two RR50 statistical power: ' + str(np.round(100*expected_NV_model_two_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model one MPC  statistical power: ' + str(np.round(100*expected_NV_model_one_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model two MPC  statistical power: ' + str(np.round(100*expected_NV_model_two_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model one TTP  statistical power: ' + str(np.round(100*expected_NV_model_one_TTP_nn_stat_power,  3)) + ' %' + \
           '\nNV model two TTP  statistical power: ' + str(np.round(100*expected_NV_model_two_TTP_nn_stat_power,  3)) + ' %' + '\n' )

    plot_predicted_statistical_powers_and_NV_model_hists(expected_NV_model_one_RR50_nn_stat_power,
                                                         expected_NV_model_two_RR50_nn_stat_power,
                                                         expected_NV_model_one_MPC_nn_stat_power,
                                                         expected_NV_model_two_MPC_nn_stat_power,
                                                         expected_NV_model_one_TTP_nn_stat_power,
                                                         expected_NV_model_two_TTP_nn_stat_power,
                                                         NV_model_one_and_two_patient_pop_hist,
                                                         monthly_mean_axis_start,
                                                         monthly_mean_axis_stop,
                                                         monthly_mean_axis_step,
                                                         monthly_mean_tick_spacing,
                                                         monthly_std_dev_axis_start,
                                                         monthly_std_dev_axis_stop,
                                                         monthly_std_dev_axis_step,
                                                         monthly_std_dev_tick_spacing)
