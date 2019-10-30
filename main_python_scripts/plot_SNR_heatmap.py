import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def retrieve_SNR_map(endpoint_name):

    with open(endpoint_name + '_SNR_data.json', 'r') as json_file:

        SNR_map = np.array(json.load(json_file))
    
    return SNR_map


def plot_power_law_curves(ax, min_power_law_slope, 
                              max_power_law_slope, 
                              power_law_slope_spacing,
                              monthly_mean_axis_start, 
                              monthly_mean_axis_stop, 
                              monthly_mean_axis_step, 
                              monthly_std_dev_axis_start, 
                              monthly_std_dev_axis_stop, 
                              monthly_std_dev_axis_step):

    monthly_mean_axis_scale_ratio = (1/monthly_mean_axis_step) + (1/(monthly_mean_axis_stop - monthly_mean_axis_start))
    monthly_std_dev_axis_scale_ratio = (1/monthly_std_dev_axis_step) + (1/(monthly_std_dev_axis_stop - monthly_std_dev_axis_start))

    monthly_mean_axis_array = np.arange(monthly_mean_axis_start, monthly_mean_axis_stop + monthly_mean_axis_step, monthly_mean_axis_step)

    power_law_slopes = np.arange(min_power_law_slope/power_law_slope_spacing, max_power_law_slope/power_law_slope_spacing + 1, 1)*power_law_slope_spacing

    for power_law_slope_index in range(len(power_law_slopes)):

        power_law_slope = power_law_slopes[power_law_slope_index]

        power_law_monthly_std_dev_axis_points = np.power(monthly_mean_axis_array, power_law_slope) + 0.5

        ax.plot(monthly_mean_axis_array*monthly_mean_axis_scale_ratio - 0.5, (monthly_std_dev_axis_stop - power_law_monthly_std_dev_axis_points)*monthly_std_dev_axis_scale_ratio)
    
    ax.legend(np.round(power_law_slopes, 1), bbox_to_anchor=(1, 0.5, 0.5, 0.5))

    return ax


def plot_SNR_map(monthly_mean_axis_start,
                 monthly_mean_axis_stop,
                 monthly_mean_axis_step,
                 monthly_mean_tick_spacing,
                 monthly_std_dev_axis_start,
                 monthly_std_dev_axis_stop,
                 monthly_std_dev_axis_step,
                 monthly_std_dev_tick_spacing,
                 min_power_law_slope,
                 max_power_law_slope,
                 power_law_slope_spacing,
                 SNR_map,
                 endpoint_name):
    
    monthly_mean_tick_labels = np.arange(monthly_mean_axis_start, monthly_mean_axis_stop + monthly_mean_tick_spacing, monthly_mean_tick_spacing)
    monthly_std_dev_tick_labels = np.arange(monthly_std_dev_axis_start, monthly_std_dev_axis_stop + monthly_std_dev_tick_spacing, monthly_std_dev_tick_spacing)

    monthly_mean_ticks = monthly_mean_tick_labels/monthly_mean_axis_step + 0.5 - 1
    monthly_std_dev_ticks = monthly_std_dev_tick_labels/monthly_std_dev_axis_step + 0.5 - 1

    monthly_std_dev_ticks = np.flip(monthly_std_dev_ticks, 0)

    fig = plt.figure()

    ax = sns.heatmap(SNR_map, cmap='RdBu_r', cbar_kws={'label':'SNR of location'})
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title(endpoint_name + ' SNR map')
    ax.set_xticks(monthly_mean_ticks)
    ax.set_xticklabels(monthly_mean_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_std_dev_ticks)
    ax.set_yticklabels(monthly_std_dev_tick_labels, rotation='horizontal')

    ax = plot_power_law_curves(ax, min_power_law_slope, 
                               max_power_law_slope, 
                               power_law_slope_spacing,
                               monthly_mean_axis_start, 
                               monthly_mean_axis_stop, 
                               monthly_mean_axis_step, 
                               monthly_std_dev_axis_start, 
                               monthly_std_dev_axis_stop, 
                               monthly_std_dev_axis_step)

    fig.savefig(endpoint_name + '_SNR_map.png', dpi=300, bbox_inches='tight')
    #import pandas as pd
    #print(pd.DataFrame(SNR_map).to_string())
    #plt.show()


if(__name__=='__main__'):

    endpoint_name = 'RR50'

    monthly_mean_axis_start   = 1
    monthly_mean_axis_stop    = 15
    monthly_mean_axis_step    = 1
    monthly_mean_tick_spacing = 1

    monthly_std_dev_axis_start   = 1
    monthly_std_dev_axis_stop    = 15
    monthly_std_dev_axis_step    = 1
    monthly_std_dev_tick_spacing = 1

    min_power_law_slope     = 0.3
    max_power_law_slope     = 1.9
    power_law_slope_spacing = 0.2

    SNR_map = retrieve_SNR_map(endpoint_name)

    plot_SNR_map(monthly_mean_axis_start,
                 monthly_mean_axis_stop,
                 monthly_mean_axis_step,
                 monthly_mean_tick_spacing,
                 monthly_std_dev_axis_start,
                 monthly_std_dev_axis_stop,
                 monthly_std_dev_axis_step,
                 monthly_std_dev_tick_spacing,
                 min_power_law_slope,
                 max_power_law_slope,
                 power_law_slope_spacing,
                 SNR_map,
                 endpoint_name)
    
