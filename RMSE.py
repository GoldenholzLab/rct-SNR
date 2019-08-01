import os
import json
import numpy as np
import sys

if(__name__=='__main__'):

    num_stat_power_estimates = int(sys.argv[1])
    folder                   = sys.argv[2]

    emp_stat_power_array          = np.zeros(num_stat_power_estimates)
    fisher_exact_stat_power_array = np.zeros(num_stat_power_estimates)

    for stat_power_estimate_index in range(num_stat_power_estimates):

        power_comp_file_path = folder + '/power_comparison_' + str(stat_power_estimate_index + 1) + '.txt'

        with open(power_comp_file_path, 'r') as text_file:
            [fisher_exact_power_str, emp_stat_power_str] = text_file.read().split(', ')
            fisher_exact_stat_power_array[stat_power_estimate_index] = float(fisher_exact_power_str)
            emp_stat_power_array[stat_power_estimate_index]          = float(emp_stat_power_str)
    
    stat_power_RMSE_str = str(np.round(np.sqrt(np.mean(np.power(emp_stat_power_array - fisher_exact_stat_power_array, 2))), 3))
    with open('RMSE.txt', 'w+') as text_file:
        text_file.write(stat_power_RMSE_str)
    
