import json
import numpy as np

def retrieve_individual_map(directory, min_req_base_sz_count, folder, file_name):

    folder_path = directory + '/' + str(min_req_base_sz_count) + '/' + folder
    file_path = folder_path + '/' + file_name

    with open(file_path, 'w+') as json_file:

        data_map = np.array(json.load(json_file))

    return map

if (__name__=='__main__'):

    