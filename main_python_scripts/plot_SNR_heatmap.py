import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sys
from PIL import Image
import io
import matplotlib.font_manager as fm


if(__name__=='__main__'):

    endpoint_names = ['RR50', 'MPC', 'TTP']
    
    max_SNR = int(sys.argv[1])
    min_SNR = int(sys.argv[2])

    monthly_mean_axis_start   = 1
    monthly_mean_axis_stop    = 15
    monthly_mean_axis_step    = 1
    monthly_mean_tick_spacing = 1

    monthly_std_dev_axis_start   = 1
    monthly_std_dev_axis_stop    = 15
    monthly_std_dev_axis_step    = 1
    monthly_std_dev_tick_spacing = 1

    power_law_slope     = 0.7

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#

    monthly_mean_tick_labels = np.arange(monthly_mean_axis_start, monthly_mean_axis_stop + monthly_mean_tick_spacing, monthly_mean_tick_spacing)
    monthly_std_dev_tick_labels = np.arange(monthly_std_dev_axis_start, monthly_std_dev_axis_stop + monthly_std_dev_tick_spacing, monthly_std_dev_tick_spacing)

    monthly_mean_ticks = monthly_mean_tick_labels/monthly_mean_axis_step + 0.5 - 1
    monthly_std_dev_ticks = monthly_std_dev_tick_labels/monthly_std_dev_axis_step + 0.5 - 1

    monthly_std_dev_ticks = np.flip(monthly_std_dev_ticks, 0)

    monthly_mean_axis_scale_ratio = (1/monthly_mean_axis_step) + (1/(monthly_mean_axis_stop - monthly_mean_axis_start))
    monthly_std_dev_axis_scale_ratio = (1/monthly_std_dev_axis_step) + (1/(monthly_std_dev_axis_stop - monthly_std_dev_axis_start))

    monthly_mean_axis_array = np.arange(monthly_mean_axis_start, monthly_mean_axis_stop + monthly_mean_axis_step, monthly_mean_axis_step)

    reg_prop  = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Regular.ttf')
    bold_prop = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Bold.ttf')

    fig = plt.figure(figsize=(20, 5))

    for endpoint_name_index in range(len(endpoint_names)):

        endpoint_name = endpoint_names[endpoint_name_index]

        with open(endpoint_name + '_SNR_data.json', 'r') as json_file:
            SNR_map = np.array(json.load(json_file))

        plt.subplot(1, 3, endpoint_name_index + 1)
        ax = sns.heatmap(SNR_map, 
                         vmin=min_SNR, 
                         vmax = max_SNR, 
                         cmap='RdBu_r',
                         cbar_kws={'format'   : '%.0f%%'},
                         mask = SNR_map == 0)
        cbar = ax.collections[0].colorbar
        ax.set_facecolor('k')
        cbar.set_label('SNR of location',                                    fontproperties=reg_prop, fontsize=14)
        ax.set_xlabel(r'monthly seizure count mean, $\mu$',                  fontproperties=reg_prop, fontsize=14)
        ax.set_ylabel(r'monthly seizure count standard deviation, $\sigma$', fontproperties=reg_prop, fontsize=14)
        ax.set_title(endpoint_name + ' SNR map',                             fontproperties=reg_prop, fontsize=15)
        ax.set_xticks(monthly_mean_ticks)
        ax.set_xticklabels(monthly_mean_tick_labels, rotation='horizontal')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        ax.set_yticks(monthly_std_dev_ticks)
        ax.set_yticklabels(monthly_std_dev_tick_labels, rotation='horizontal')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        ax.text(-0.25, 1, string.ascii_uppercase[endpoint_name_index] + ')', 
                fontproperties=bold_prop, transform=ax.transAxes, size=20)
        
        power_law_monthly_std_dev_axis_points = np.power(monthly_mean_axis_array, power_law_slope) + 0.5

        ax.plot( monthly_mean_axis_array*monthly_mean_axis_scale_ratio - 0.5, 
                 (monthly_std_dev_axis_stop - power_law_monthly_std_dev_axis_points)*monthly_std_dev_axis_scale_ratio,
                 color='green')
    
    plt.subplots_adjust(wspace = .25)

    #fig.savefig(endpoint_name + '_SNR_map.png', dpi=600, bbox_inches='tight')
    #import pandas as pd
    #print(pd.DataFrame(SNR_map).to_string())
    #plt.show()

    png1 = io.BytesIO()
    fig.savefig(png1, dpi = 600, bbox_inches = 'tight', format = 'png')
    png2 = Image.open(png1)
    png2.save('Romero-fig3.tiff')
    png1.close()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------#
