import os
import json
import numpy as np
import matplotlib.pyplot as plt


if(__name__=='__main__'):

    num_stat_power_estimates = 1000
    bins = 100
    folder = '/Users/juanromero/Documents/Python_3_Files/useless_folder'

    empirical_stat_power_list = []
    semi_ana_stat_power_list  = []
    ana_stat_power_list       = []    

    for stat_power_index in range(num_stat_power_estimates):

        file_path = folder + '/' + str(stat_power_index) + '.json'

        if( os.path.isfile(file_path) ):
            with open(file_path, 'r') as json_file:
                stat_power_array = json.load(json_file)

                empirical_stat_power_list.append(stat_power_array[0])
                semi_ana_stat_power_list.append(stat_power_array[1])
                ana_stat_power_list.append(stat_power_array[2])

    empirical_stat_power_array = np.array(empirical_stat_power_list)
    semi_ana_stat_power_array  = np.array(semi_ana_stat_power_list)
    ana_stat_power_array       = np.array(ana_stat_power_list)

    semi_ana_stat_power_error_array = empirical_stat_power_array - semi_ana_stat_power_array
    ana_stat_power_error_array      = empirical_stat_power_array - ana_stat_power_array

    semi_ana_stat_power_mean_error = np.mean(semi_ana_stat_power_error_array)
    ana_stat_power_mean_error      = np.mean(ana_stat_power_error_array)
    semi_ana_stat_power_RMSE       = np.sqrt(np.mean(np.power(semi_ana_stat_power_error_array, 2)))
    ana_stat_power_RMSE            = np.sqrt(np.mean(np.power(ana_stat_power_error_array,      2)))

    semi_ana_stat_power_mean_error_str = str(np.round(semi_ana_stat_power_mean_error, 3))
    semi_ana_stat_power_RMSE_str       = str(np.round(semi_ana_stat_power_RMSE, 3))
    semi_ana_stat_power_error_str      = semi_ana_stat_power_mean_error_str + ' ± ' + semi_ana_stat_power_RMSE_str

    ana_stat_power_mean_error_str = str(np.round(ana_stat_power_mean_error, 3))
    ana_stat_power_RMSE_str       = str(np.round(ana_stat_power_RMSE, 3))
    ana_stat_power_error_str            = ana_stat_power_mean_error_str + ' ± ' + ana_stat_power_RMSE_str

    stat_power_error_str = 'semi-analytical statistical power error:      ' + semi_ana_stat_power_error_str + '\n' + \
                           'map-based analytical statistical power error: ' + ana_stat_power_error_str
    
    text_file_name = 'TTP_stat_power_error'
    text_file_path = os.getcwd() + '/' + text_file_name + '.txt'
    with open(text_file_path, 'w+') as text_file:
        text_file.write(stat_power_error_str)

    plt.figure()
    plt.hist(semi_ana_stat_power_error_array, bins=bins)
    plt.title('semi-analytical statistical power estimate error histogram')
    plt.xlabel('numerical difference in statistical power estimates (relative to empirical)')
    plt.ylabel('frequency of error')
    plt.savefig(os.getcwd() + '/semi_ana_stat_power_error.png')

    plt.figure()
    plt.hist(ana_stat_power_error_array, bins=bins)
    plt.title('analytical statistical power estimate error histogram')
    plt.xlabel('numerical difference in statistical power estimates (relative to empirical)')
    plt.ylabel('frequency of error')
    plt.savefig(os.getcwd() + '/ana_stat_power_error.png')
