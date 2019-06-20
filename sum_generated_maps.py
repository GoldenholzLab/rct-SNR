import json
import numpy as np
import os


def average_map_type(directory,         min_req_base_sz_count, endpoint_statistic_map_file_name, 
                     num_monthly_means, num_monthly_std_devs,  num_maps):
    '''

    This function retrieves multiple data maps (2D Numpy array), all of the same endpoint statistic, from 
    
    intermediate JSON files. Each point on one data map is calculated from a few simulated trials. After collecting 
    
    these undersampled maps, this function takes the average of the maps in order to create one map where each
    
    point is properly sampled by a large number of trials.

    Inputs:

        1) directory:

            (string) - the name of the directory which contains the folder in which the 
                       
                       intermediate JSON files are stored
        
        2) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts

        3) endpoint_statistic_map_file_name:

            (string) - the name of the JSON file containing an undersampled endpoint statistic map

        4) num_monthly_means:
        
            (int) - the number of ticks on the monthly seizure count mean axis

        5) num_monthly_std_devs:

            (int) - the number of ticks on the monthly seizure count standard deviation axis
        
        6) num_maps:

            (int) - the number of undersampled maps to average over

    Outputs:

        1) average_endpoint_statistic_map:

            (2D Numpy array) - a properly sampled endpoint statistic map, calculated from an average
                               
                               of undersampled endpoint statistic maps

    '''

    # initialize the 2D Numpy array which will contain the average endpoint statistic map
    average_endpoint_statistic_map = np.zeros((num_monthly_std_devs, num_monthly_means))

    # for each folder containing an undersampled endpoint statistic map
    for folder_num in range(1, num_maps):
        
        # locate the path of the folder in which the JSON file of the map is stored
        folder_path = directory + '/' + str(min_req_base_sz_count) + '/' + str(folder_num)

        # locate the actual file path of the JSON file based on the folder path and the file name
        endpoint_statistic_map_file_path = folder_path + '/' + endpoint_statistic_map_file_name + '.json'

        # open the JSON file
        with open(endpoint_statistic_map_file_path, 'r') as endpoint_statistic_map_json_file:

            # convert it to a 2D Numpy array
            endpoint_statistic_map = np.array(json.load(endpoint_statistic_map_json_file))

        # add the current map to the final average map
        average_endpoint_statistic_map = average_endpoint_statistic_map + endpoint_statistic_map

    # average the endpoint statistic map
    average_endpoint_statistic_map = average_endpoint_statistic_map/len(range(1, num_maps))

    return average_endpoint_statistic_map



if (__name__=='__main__'):

    directory = '/Users/juanromero/Documents/GitHub/rct-SNR'
    min_req_base_sz_count = 0
    endpoint_statistic_map_file_name = 'TTP_type_1_error_map'

    '''
    data_map_file_names = ['expected_placebo_RR50_map', 'expected_placebo_MPC_map', 'expected_placebo_TTP_map',
                           'expected_drug_RR50_map',    'expected_drug_MPC_map',    'expected_drug_TTP_map',
                           'RR50_stat_power_map',       'MPC_stat_power_map',       'TTP_stat_power_map',
                           'RR50_type_1_error_map',     'MPC_type_1_error_map',     'TTP_type_1_error_map']
    '''

    meta_data_file_name = 'meta_data.txt'
    meta_data_file_path = directory + '/' + meta_data_file_name

    with open( meta_data_file_path, 'r' ) as meta_data_text_file:

        start_monthly_mean = int( meta_data_text_file.readline() )
        stop_monthly_mean = int( meta_data_text_file.readline() )
        step_monthly_mean = int( meta_data_text_file.readline() )
        start_monthly_std_dev = int( meta_data_text_file.readline() )
        stop_monthly_std_dev = int( meta_data_text_file.readline() )
        step_monthly_std_dev = int( meta_data_text_file.readline() )
        num_maps = int( meta_data_text_file.readline() )

    num_monthly_means = int( (stop_monthly_mean - start_monthly_mean)/step_monthly_mean ) + 1
    num_monthly_std_devs = int( (stop_monthly_std_dev - start_monthly_std_dev)/step_monthly_std_dev ) + 1

    average_endpoint_statistic_map = \
        average_map_type(directory,         min_req_base_sz_count, endpoint_statistic_map_file_name, 
                         num_monthly_means, num_monthly_std_devs,  num_maps)

    folder_path = directory + '/final'
    average_endpoint_statistic_file_path = folder_path + '/' + endpoint_statistic_map_file_name + '.json'

    if( not os.path.exists(folder_path) ):

        os.makedirs(folder_path)
    
    with open(average_endpoint_statistic_file_path, 'w+') as json_file:

        json.dump(average_endpoint_statistic_map.tolist(), average_endpoint_statistic_file_path)    


    
