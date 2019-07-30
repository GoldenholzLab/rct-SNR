import sys
import os
import json
import numpy as np

if (__name__=='__main__'):

    min_percentile = int(sys.argv[1])
    max_percentile = int(sys.argv[2])
    num_iter       = int(sys.argv[3])
    folder         = sys.argv[4]

    fisher_exact_emp_stat_power_RMSE_array = np.zeros(max_percentile + 1 - min_percentile)
    fisher_exact_ana_stat_power_RMSE_array = np.zeros(max_percentile + 1 - min_percentile)
    data_str = ''

    for percentile in range(min_percentile, max_percentile + 1):
        
        fisher_exact_emp_stat_power_err_sq_array_per_percentile = []
        fisher_exact_ana_stat_power_err_sq_array_per_percentile = []

        for iter_num in range(1, num_iter + 1):

            json_filename = 'percentile_' + str(percentile) + '_|_iteration' + '_' + str(iter_num) + '.json' 
            json_filepath = folder + '/' + json_filename

            if( os.path.isfile(json_filepath) ):

                with open(json_filepath, 'r') as json_file:

                    data = np.array(json.load(json_file))
        
                RR50_emp_stat_power         = data[0]
                fisher_exact_emp_stat_power = data[1]
                fisher_exact_ana_stat_power = data[2]

                fisher_exact_emp_stat_power_err_sq_array_per_percentile.append(np.power(RR50_emp_stat_power - fisher_exact_emp_stat_power, 2))
                fisher_exact_ana_stat_power_err_sq_array_per_percentile.append(np.power(RR50_emp_stat_power - fisher_exact_ana_stat_power, 2))
        
        fisher_exact_emp_stat_power_err_sq_array_per_percentile = np.array(fisher_exact_emp_stat_power_err_sq_array_per_percentile)
        fisher_exact_ana_stat_power_err_sq_array_per_percentile = np.array(fisher_exact_ana_stat_power_err_sq_array_per_percentile)

        fisher_exact_emp_stat_power_RMSE_per_percentile = np.round(np.sqrt(np.mean(fisher_exact_emp_stat_power_err_sq_array_per_percentile)), 3)
        fisher_exact_ana_stat_power_RMSE_per_percentile = np.round(np.sqrt(np.mean(fisher_exact_ana_stat_power_err_sq_array_per_percentile)), 3)
    
        data_str = data_str + '\n[percentile, RMSE]: [' + str(percentile) + ', ' + str(fisher_exact_emp_stat_power_RMSE_per_percentile) + ', ' + str(fisher_exact_ana_stat_power_RMSE_per_percentile) + ']'
    
    data_str = data_str + '\n'

    text_filename = 'final_data.txt'
    text_filepath = folder + '/' + text_filename
    with open(text_filepath, 'w+') as text_file:
        text_file.write(data_str)
