import sys
import os
import json
import numpy as np

if (__name__=='__main__'):

    num_iter = int(sys.argv[1])

    fisher_exact_emp_stat_power_err_sq_array = []
    fisher_exact_ana_stat_power_err_sq_array = []

    for iter_num in range(1, num_iter + 1):

        json_filepath = os.getcwd() + '/' + str(iter_num) + '.json'
        if ( os.path.isfile(json_filepath) ):
            with open(json_filepath, 'r') as json_file:
                data = json.load(json_file)
    
            RR50_emp_stat_power         = data[0]
            fisher_exact_emp_stat_power = data[1]
            fisher_exact_ana_stat_power = data[2]

            fisher_exact_emp_stat_power_err = RR50_emp_stat_power - fisher_exact_emp_stat_power
            fisher_exact_ana_stat_power_err = RR50_emp_stat_power - fisher_exact_ana_stat_power
    
            fisher_exact_emp_stat_power_err_sq = np.power(fisher_exact_emp_stat_power_err, 2)
            fisher_exact_ana_stat_power_err_sq = np.power(fisher_exact_ana_stat_power_err, 2)

            fisher_exact_emp_stat_power_err_sq_array.append(fisher_exact_emp_stat_power_err_sq)
            fisher_exact_ana_stat_power_err_sq_array.append(fisher_exact_ana_stat_power_err_sq)

    fisher_exact_emp_stat_power_RMSE = np.sqrt(np.mean(fisher_exact_emp_stat_power_err_sq_array))
    fisher_exact_ana_stat_power_RMSE = np.sqrt(np.mean(fisher_exact_ana_stat_power_err_sq_array))

    fisher_exact_emp_stat_power_RMSE_str = str(np.round(fisher_exact_emp_stat_power_RMSE, 3))
    fisher_exact_ana_stat_power_RMSE_str = str(np.round(fisher_exact_ana_stat_power_RMSE, 3))

    print(fisher_exact_emp_stat_power_RMSE_str)
    print(fisher_exact_ana_stat_power_RMSE_str)
