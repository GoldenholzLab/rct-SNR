import json
import numpy as np
import os


def calculate_average_endpoint_statistic_map(directory,         min_req_base_sz_count, endpoint_statistic_map_file_name, 
                                             num_monthly_means, num_monthly_std_devs,  num_maps):
    '''

    This function retrieves multiple undersampled data maps (2D Numpy arrays), all of the same endpoint statistic, from 
    
    intermediate JSON files. Every point on one endpoint statistic map is calculated from a few simulated trials (thus the 
    
    undersampling). Each undersampled endpoint statistic map is spread across a specific file tree structure, of which a 
    
    simplified version is shown below:


                            directory
                            |
                            +---min_req_base_sz_count (e.g., 1)
                            |   |
                            |   +---folder_num (# 1)
                            |   |   |
                            |   |   +---endpoint_statistic_map_file_name.json (e.g., expected median percent change placebo response)
                            |   |   |
                            |   |   \---endpoint_statistic_map_file_name.json (e.g., 50% responder rate statistical power)
                            |   |
                            |   \---folder_num (# 2)
                            |       |
                            |       +---endpoint_statistic_map_file_name.json (e.g., expected median percent change placebo response)
                            |       |
                            |       \---endpoint_statistic_map_file_name.json (e.g., 50% responder rate statistical power)
                            |
                            \---min_req_base_sz_count (e.g., 2)
                                |
                                +---folder_num (# 1)
                                |   |
                                |   +---endpoint_statistic_map_file_name.json (e.g., expected median percent change placebo response)
                                |   |
                                |   \---endpoint_statistic_map_file_name.json (e.g., 50% responder rate statistical power)
                                |
                                \---folder_num (# 2)
                                    |
                                    +---endpoint_statistic_map_file_name.json (e.g., expected median percent change placebo response)
                                    |
                                    \---endpoint_statistic_map_file_name.json (e.g., 50% responder rate statistical power)
    

    After collecting these undersampled maps, the average of the maps is calculated in order to create one endpoint 
    
    statistic map where each point is properly sampled by a large number of trials. Finally, the average map for the 
    
    endpoint statistic is stored in a folder called 'final' on the branch as 'folder_num'. The average endpoint statistic map
    
    map is not calculated across the min_req_base_sz_count parameter: for each different value of min_req_base_sz_count,

    there is a different average endpoint statistic map (and accordingly, a different 'final' folder for each min_req_base_sz_count
    
    as well).

    Inputs:

        1) directory:

            (string) - the name of the directory which contains the folder in which the 
                       
                       intermediate JSON files are stored
        
        2) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts

        3) endpoint_statistic_map_file_name:

            (string) - the typical name of the JSON file containing an undersampled endpoint statistic map

        4) num_monthly_means:
        
            (int) - the number of ticks on the monthly seizure count mean axis

        5) num_monthly_std_devs:

            (int) - the number of ticks on the monthly seizure count standard deviation axis
        
        6) num_maps:

            (int) - the number of undersampled maps to average over

    Outputs:

        Technically None

    '''

    # initialize the 2D Numpy array which will contain the average endpoint statistic map
    average_endpoint_statistic_map = np.zeros((num_monthly_std_devs, num_monthly_means))

    # for each folder containing an undersampled endpoint statistic map
    for folder_num in range(1, num_maps):
        
        # locate the path of the folder in which the JSON file of the map is stored
        folder_num_path = directory + '/' + str(min_req_base_sz_count) + '/' + str(folder_num)

        # locate the actual file path of the JSON file based on the folder path and the file name
        endpoint_statistic_map_file_path = folder_num_path + '/' + endpoint_statistic_map_file_name + '.json'

        # open the JSON file
        with open(endpoint_statistic_map_file_path, 'r') as endpoint_statistic_map_json_file:

            # convert it to a 2D Numpy array
            endpoint_statistic_map = np.array(json.load(endpoint_statistic_map_json_file))

        # add the current map to the final average map
        average_endpoint_statistic_map = average_endpoint_statistic_map + endpoint_statistic_map

    # average the endpoint statistic map
    average_endpoint_statistic_map = average_endpoint_statistic_map/len(range(1, num_maps))

    # create the file path of the folder that the average endpoint statistic map will be stored, as well as the file path for that same map
    average_map_folder_path = directory + '/final'
    average_endpoint_statistic_map_file_path = average_map_folder_path + '/' + endpoint_statistic_map_file_name + '.json'

    # if the final folder for the average map doesn't exist, create it
    if( os.path.exists(average_map_folder_path) ):

        os.makedirs(average_map_folder_path)

    # store the average endpoint statistic map into a JSON file.
    with open(average_endpoint_statistic_map_file_path, 'w+') as json_file:

        json.dump(average_endpoint_statistic_map.tolist(), json_file)



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

    calculate_average_endpoint_statistic_map(directory,         min_req_base_sz_count, endpoint_statistic_map_file_name, 
                                             num_monthly_means, num_monthly_std_devs,  num_maps)



    
