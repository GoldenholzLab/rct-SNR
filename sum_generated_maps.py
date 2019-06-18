import json
import numpy as np

def retrieve_individual_map(directory, min_req_base_sz_count, folder, file_name):
    '''

    

    '''
    folder_path = directory + '/' + str(min_req_base_sz_count) + '/' + str(folder)
    file_path = folder_path + '/' + file_name + '.json'

    with open(file_path, 'r') as json_file:

        data_map = np.array(json.load(json_file))

    return data_map


if (__name__=='__main__'):

    directory = '/Users/juanromero/Documents/GitHub/rct-SNR'
    min_req_base_sz_count = 0
    folder = 1
    file_name = 'MPC_stat_power_map'

    print( retrieve_individual_map(directory, min_req_base_sz_count, folder, file_name) )