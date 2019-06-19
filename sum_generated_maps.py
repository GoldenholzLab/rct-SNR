import json
import numpy as np
import pandas as pd

def retrieve_individual_map(directory, min_req_base_sz_count, folder, data_map_file_name):
    '''

    This function retrieves one data map as it was stored in an intermediate JSON

    file.

    Inputs:

        1) directory:

            (string) - the name of the directory which contains the folder in which 
                       
                       the map is stored as a JSON file

        2) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        3) folder:

            (string) - the name of the actual folder in which all the intermediate JSON files 
                       
                       will be stored
        
        4) data_map_file_name:

            (string) - the file name of the JSON file in which the map is stored as an intermediate
                       
                       JSON file

    Outputs:

        1) data_map:

            (2D Numpy array) - the actual map which was stored in the JSON file

    '''

    # locate the path of the folder in which the JSON file is stored
    folder_path = directory + '/' + str(min_req_base_sz_count) + '/' + folder

    # locate the actual file path of the JSON file based on the folder path and the file name
    data_map_file_path = folder_path + '/' + data_map_file_name + '.json'

    # open the JSON file
    with open(data_map_file_path, 'r') as data_map_json_file:

        # convert it to a 2D Numpy array
        data_map = np.array(json.load(data_map_json_file))

    return data_map


if (__name__=='__main__'):

    directory = '/Users/juanromero/Documents/GitHub/rct-SNR'
    min_req_base_sz_count = 0
    data_map_file_name = 'MPC_stat_power_map'

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

    final_map = np.zeros((num_monthly_std_devs, num_monthly_means))

    for folder_num in range(1, num_maps):
        
        data_map = retrieve_individual_map(directory, min_req_base_sz_count, str(folder_num), data_map_file_name)

        final_map = final_map + data_map

        print( pd.DataFrame( final_map ).to_string() + '\n\n' )
    
    final_map = final_map/len(range(1, num_maps))

    print( pd.DataFrame( final_map ).to_string() + '\n\n' )

    
    
