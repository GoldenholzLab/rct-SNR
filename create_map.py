import numpy as np
import json

directory = '/Users/juanromero/Documents/python_3_Files/test'
min_req_base_sz_count = 0
map_num = 1
monthly_mean = 4
monthly_std_dev = 3

min_req_base_sz_count_str = str(min_req_base_sz_count)
map_num_str = str(map_num)
monthly_mean_str = str(np.float64(monthly_mean))
monthly_std_dev_str = str(np.float64(monthly_std_dev))

file_path = directory + '/eligibility_criteria__' + min_req_base_sz_count_str + '/map_num__' + \
            map_num_str + '/mean__'  + monthly_mean_str + '/std_dev__' + monthly_std_dev_str + '.json'

with open(file_path, 'r') as json_file:

    endpoint_statistics = np.array(json.load(json_file))

    
