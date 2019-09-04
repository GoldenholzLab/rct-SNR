import sys
import os
import json
import numpy as np

if(__name__=='__main__'):

    num_stat_power_estimates = int(sys.argv[1])
    folder                   =     sys.argv[2]

    coxph_stat_power_list = []
    TTP_emp_stat_power_list = []

    for stat_power_estimate_index in range(num_stat_power_estimates):

        file_name = str(stat_power_estimate_index) + 1
        file_path = folder + '/' + file_name + '.json'

        if( os.path.isfile(file_path) ):
            with open(file_path, 'r') as json_file:
                [copxh_stat_power, TTP_emp_stat_power] = json.load(json_file)
                coxph_stat_power_list.append(copxh_stat_power)
                TTP_emp_stat_power_list.append(TTP_emp_stat_power)
    
    coxph_stat_power_array = np.array(coxph_stat_power_list)
    TTP_emp_stat_power_array = np.array(TTP_emp_stat_power_list)
    TTP_error_array = TTP_emp_stat_power_array - coxph_stat_power_array

    TTP_stat_power_RMSE_str = str(np.round(np.sqrt(np.mean(np.power(TTP_error_array, 2))), 3))
    print(TTP_stat_power_RMSE_str)
