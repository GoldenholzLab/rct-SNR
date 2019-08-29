import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt

if(__name__=='__main__'):

    num_stat_power_estimates = int(sys.argv[1])
    folder                   = sys.argv[2]

    fisher_exact_stat_power_list = []
    emp_stat_power_list = []

    for stat_power_estimate_index in range(num_stat_power_estimates):

        power_comp_file_path = folder + '/power_comparison_' + str(stat_power_estimate_index + 1) + '.txt'

        if( os.path.isfile(power_comp_file_path) ):
            with open(power_comp_file_path, 'r') as text_file:
                [fisher_exact_power_str, emp_stat_power_str] = text_file.read().split(', ')
                fisher_exact_stat_power_list.append(float(fisher_exact_power_str))
                emp_stat_power_list.append(float(emp_stat_power_str))
    
    fisher_exact_stat_power_array = np.array(fisher_exact_stat_power_list)
    emp_stat_power_array          = np.array(emp_stat_power_list)
    error_array                   = emp_stat_power_array - fisher_exact_stat_power_array

    stat_power_RMSE_str = str(np.round(np.sqrt(np.mean(np.power(error_array, 2))), 3))
    with open('RMSE.txt', 'w+') as text_file:
        text_file.write(stat_power_RMSE_str)
    
    plt.figure()
    plt.hist(error_array, bins=50, density=True)
    plt.ylabel('frequency of errors')
    plt.xlabel('analytical statistical power estimate errors')
    plt.title('histogram of Map_based Fisher Exact statistical power estimate errors')
    plt.savefig('histogram_of_statistical_power_estimate_errors.png')