import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle as rectangle
import seaborn as sns
import string
from PIL import Image
import io
import sys
import matplotlib.font_manager as fm


def retrieve_expected_responses_and_hist(expected_NV_model_responses_file_name,
                                         NV_model_hist_file_name):

    with open(expected_NV_model_responses_file_name + '.json', 'r') as json_file:

        [expected_NV_model_one_RR50_nn_stat_power, 
         expected_NV_model_one_MPC_nn_stat_power, 
         expected_NV_model_one_TTP_nn_stat_power,
         expected_NV_model_two_RR50_nn_stat_power, 
         expected_NV_model_two_MPC_nn_stat_power, 
         expected_NV_model_two_TTP_nn_stat_power] = json.load(json_file)
        
    with open(NV_model_hist_file_name + '.json', 'r') as json_file:

        NV_model_one_and_two_patient_pop_hist = np.array(json.load(json_file))
    
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
                                                         monthly_mean_min,
                                                         monthly_mean_max,
                                                         monthly_std_dev_min,
                                                         monthly_std_dev_max):

    print( '\nNV model one RR50 statistical power: ' + str(np.round(100*expected_NV_model_one_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model two RR50 statistical power: ' + str(np.round(100*expected_NV_model_two_RR50_nn_stat_power, 3)) + ' %' + \
           '\nNV model one MPC  statistical power: ' + str(np.round(100*expected_NV_model_one_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model two MPC  statistical power: ' + str(np.round(100*expected_NV_model_two_MPC_nn_stat_power,  3)) + ' %' + \
           '\nNV model one TTP  statistical power: ' + str(np.round(100*expected_NV_model_one_TTP_nn_stat_power,  3)) + ' %' + \
           '\nNV model two TTP  statistical power: ' + str(np.round(100*expected_NV_model_two_TTP_nn_stat_power,  3)) + ' %' + '\n' )

    monthly_mean_axis_start   = monthly_mean_min
    monthly_mean_axis_stop    = monthly_mean_max - 1
    monthly_mean_axis_step    = 1
    monthly_mean_tick_spacing = 1

    monthly_std_dev_axis_start   = monthly_std_dev_min
    monthly_std_dev_axis_stop    = monthly_std_dev_max - 1
    monthly_std_dev_axis_step    = 1
    monthly_std_dev_tick_spacing = 1

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

    reg_prop  = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Regular.ttf')
    bold_prop = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Bold.ttf')

    fig = plt.figure(figsize=(20, 6))

    ax = plt.subplot(1,2,1)

    ax = sns.heatmap(NV_model_one_and_two_patient_pop_hist, cmap='RdBu_r', cbar_kws={'label':'Number of patients'})
    cbar = ax.collections[0].colorbar
    ax.set_xticks(monthly_mean_ticks)
    ax.set_xticklabels(monthly_mean_tick_labels, rotation='horizontal')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    ax.set_yticks(monthly_std_dev_ticks)
    ax.set_yticklabels(monthly_std_dev_tick_labels, rotation='horizontal')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    cbar.set_label('SNR of location',                                    fontproperties=reg_prop, fontsize=14)
    ax.set_xlabel(r'monthly seizure count mean, $\mu$',                  fontproperties=reg_prop, fontsize=14)
    ax.set_ylabel(r'monthly seizure count standard deviation, $\sigma$', fontproperties=reg_prop, fontsize=14)
    ax.title.set_text('2D histograms of patients from NV model 1 and 2')
    ax.text(-0.2, 1, string.ascii_uppercase[0] + ')', 
            fontproperties=bold_prop, transform=ax.transAxes, size=20)
    
    NV_model_1_rect = rectangle((0.5, 11.5), 5, 3, 0, edgecolor='xkcd:apple green', facecolor='none', linewidth=3)
    NV_model_2_rect = rectangle((5.5,  9.5), 7, 4, 0, edgecolor='xkcd:vivid green', facecolor='none', linewidth=3)
    ax.add_artist(NV_model_1_rect)
    ax.add_artist(NV_model_2_rect)
    ax.text(0.6, 11.2, 
            'NV Model 1', 
            fontproperties=reg_prop,
            fontsize=16,
            color='white')
    ax.text(8.6, 9.2, 
            'NV Model 2', 
            fontproperties=reg_prop,
            fontsize=16,
            color='white')
    '''
    fontdict = {'family': 'serif',
                        'color':  'white',
                        'weight': 'normal',
                        'size': 16}   
    '''
    #ax.legend([NV_model_1_rect, NV_model_2_rect], ['NV Model 1', 'NV Model 2'], loc='upper left')

    plt.subplot(1,2,2)

    plt.bar(index, 
            NV_model_one_endpoint_stat_powers, 
            bar_width,
            alpha=opacity,
            color='b',
            label='NV Model one')

    plt.bar(index + bar_width, 
            NV_model_two_endpoint_stat_powers, 
            bar_width,
            alpha=opacity,
            color='g',
            label='NV Model two')

    plt.xlabel('endpoint',                                  fontproperties=reg_prop, fontsize=14)
    plt.ylabel('statistical power',                         fontproperties=reg_prop, fontsize=14)
    plt.title('average predicted statistical powers',       fontproperties=reg_prop, fontsize=14)
    plt.xticks(index + bar_width/2, ('RR50', 'MPC', 'TTP'), fontproperties=reg_prop, fontsize=14)
    plt.yticks(np.arange(0, 110, 10))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.yaxis.grid(True)
    plt.legend(prop=reg_prop)
    ax.text(-0.2, 1, string.ascii_uppercase[1] + ')', 
            fontproperties=bold_prop, transform=ax.transAxes, size=20, weight='bold')

    plt.subplots_adjust(wspace = .25)

    png1 = io.BytesIO()
    fig.savefig(png1, dpi = 600, bbox_inches = 'tight', format = 'png')
    png2 = Image.open(png1)
    png2.save('Romero-fig4.tiff')
    png1.close()


def take_inputs_from_command_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    expected_NV_model_responses_file_name = sys.argv[5]
    NV_model_hist_file_name = sys.argv[6]

    return [monthly_mean_min,
            monthly_mean_max,
            monthly_std_dev_min,
            monthly_std_dev_max,
            expected_NV_model_responses_file_name,
            NV_model_hist_file_name]


if(__name__=='__main__'):

    [monthly_mean_min,
     monthly_mean_max,
     monthly_std_dev_min,
     monthly_std_dev_max,
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
         retrieve_expected_responses_and_hist(expected_NV_model_responses_file_name,
                                              NV_model_hist_file_name)

    plot_predicted_statistical_powers_and_NV_model_hists(expected_NV_model_one_RR50_nn_stat_power,
                                                         expected_NV_model_two_RR50_nn_stat_power,
                                                         expected_NV_model_one_MPC_nn_stat_power,
                                                         expected_NV_model_two_MPC_nn_stat_power,
                                                         expected_NV_model_one_TTP_nn_stat_power,
                                                         expected_NV_model_two_TTP_nn_stat_power,
                                                         NV_model_one_and_two_patient_pop_hist,
                                                         monthly_mean_min,
                                                         monthly_mean_max,
                                                         monthly_std_dev_min,
                                                         monthly_std_dev_max)

