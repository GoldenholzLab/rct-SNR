import numpy as np
import json
import sys

if (__name__=='__main__'):

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])
    folder              =     sys.argv[5]

    num_monthly_means     = monthly_mean_max - monthly_mean_min + 1
    num_monthly_std_devs  = monthly_std_dev_max - monthly_std_dev_min + 1
    monthly_mean_array    = np.arange(monthly_mean_min,    monthly_mean_max + 1)
    monthly_std_dev_array = np.arange(monthly_std_dev_min, monthly_std_dev_max + 1)
    
    average_postulated_log_hazard_ratio_map_file_name = 'average_postulated_log_hazard_ratio_map'
    average_prob_fail_placebo_arm_map_file_name       = 'average_prob_fail_placebo_arm_map'
    average_prob_fail_drug_arm_map_file_name          = 'average_prob_fail_drug_arm_map'

    average_postulated_log_hazard_ratio_map_file_path = folder + '/' + average_postulated_log_hazard_ratio_map_file_name + '.json'
    average_prob_fail_placebo_arm_map_file_path       = folder + '/' + average_prob_fail_placebo_arm_map_file_name + '.json'
    average_prob_fail_drug_arm_map_file_path          = folder + '/' + average_prob_fail_drug_arm_map_file_name + '.json'
    
    average_postulated_log_hazard_ratio_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    average_prob_fail_placebo_arm_map       = np.zeros((num_monthly_std_devs, num_monthly_means))
    average_prob_fail_drug_arm_map          = np.zeros((num_monthly_std_devs, num_monthly_means))
  
    for monthly_mean in monthly_mean_array:
        for monthly_std_dev in monthly_std_dev_array:

            print([monthly_mean, monthly_std_dev])

            json_file_name = str(monthly_mean) + '_' + str(monthly_std_dev)
            json_file_path = folder + '/' + json_file_name + '.json'
            with open(json_file_path, 'r') as json_file:
                map_point_data_list = json.load(json_file)
                average_postulated_log_hazard_ratio = map_point_data_list[0]
                average_prob_fail_placebo_arm       = map_point_data_list[1]
                average_prob_fail_drug_arm          = map_point_data_list[2]
            
            average_postulated_log_hazard_ratio_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min] = average_postulated_log_hazard_ratio
            average_prob_fail_placebo_arm_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min]       = average_prob_fail_placebo_arm
            average_prob_fail_drug_arm_map[monthly_std_dev_max - monthly_std_dev, monthly_mean - monthly_mean_min]          = average_prob_fail_drug_arm

    with open(average_postulated_log_hazard_ratio_map_file_path, 'w+') as json_file:
        json.dump(average_postulated_log_hazard_ratio_map.tolist(), json_file)
    
    with open(average_prob_fail_placebo_arm_map_file_path, 'w+') as json_file:
        json.dump(average_prob_fail_placebo_arm_map.tolist(), json_file)
    
    with open(average_prob_fail_drug_arm_map_file_path, 'w+') as json_file:
        json.dump(average_prob_fail_drug_arm_map.tolist(), json_file)


    import pandas as pd
    data_str = 'map of average log hazard ratio\n\n'                       + pd.DataFrame(average_postulated_log_hazard_ratio_map).to_string() + '\n\n' + \
               'map of average probability of failure in placebo arm:\n\n' + pd.DataFrame(100*average_prob_fail_placebo_arm_map).to_string()   + '\n\n' + \
               'map of average probability of failure in drug arm:\n\n'    + pd.DataFrame(100*average_prob_fail_drug_arm_map).to_string()
    print('\n' + data_str + '\n')
    
