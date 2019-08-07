import numpy as np
import sys
import json
import os

def store_results(monthly_mean,
                  monthly_std_dev,
                  average_postulated_log_hazard_ratio, 
                  average_prob_fail_placebo_arm, 
                  average_prob_fail_drug_arm,
                  folder):

    json_file_name = str(monthly_mean) + '_' + str(monthly_std_dev)
    json_file_path = folder + '/' + json_file_name + '.json'
    if ( not os.path.isdir(folder) ):
        os.makedirs(folder)
    with open(json_file_path, 'w+') as json_file:
        json.dump([average_postulated_log_hazard_ratio, average_prob_fail_placebo_arm, average_prob_fail_drug_arm], json_file)

if (__name__=='__main__'):

    monthly_mean    = int(sys.argv[1])
    monthly_std_dev = int(sys.argv[2])
    folder          =     sys.argv[3]

    print([monthly_mean, monthly_std_dev])
    
    store_results(monthly_mean,
                  monthly_std_dev,
                  np.nan, 
                  np.nan, 
                  np.nan,
                  folder)

    print(str([np.nan, np.nan, np.nan]) + '\n')