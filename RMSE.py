import os
import json
import numpy as np
import sys

if(__name__=='__main__'):

    num_iter_iter = int(sys.argv[1])
    emp_stat_power_array = np.array([])
    fisher_exact_stat_power_array = np.array([])

    for iter_iter_index in range(1, num_iter_iter+1):

        emp_file_path = os.getcwd() + '/emp_power_array_' + str(iter_iter_index) + '.json'
        map_file_path = os.getcwd() + '/fisher_exact_power_array_' + str(iter_iter_index) + '.json'

        with open(emp_file_path, 'r') as json_file:
            emp_stat_power_array = np.append(emp_stat_power_array, np.array(json.load(json_file)))
        with open(map_file_path, 'r') as json_file:
            fisher_exact_stat_power_array = np.append(fisher_exact_stat_power_array, np.array(json.load(json_file)))
        
    stat_power_RMSE_str = str(np.round(np.sqrt(np.mean(np.power(emp_stat_power_array - fisher_exact_stat_power_array, 2))), 3))
    with open('RMSE.txt', 'w+') as text_file:
        text_file.write(stat_power_RMSE_str)
