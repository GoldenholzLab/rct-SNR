import json
import numpy as np
import os
import sys


def calculate_average_endpoint_statistic_map(directory, min_req_base_sz_count, endpoint_statistic_map_file_name, 
                                             num_monthly_means, num_monthly_std_devs,  num_maps):
    '''

    Purpose:

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

            (string) - the name of the directory which contains the folders in which the intermediate JSON files are stored
        
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
    for folder_num in range(1, num_maps + 1):
        
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
    average_map_folder_path = directory + '/' + str(min_req_base_sz_count) + '/final'
    average_endpoint_statistic_map_file_path = average_map_folder_path + '/' + endpoint_statistic_map_file_name + '.json'

    # if the final folder for the average map doesn't exist, create it
    if( not os.path.exists(average_map_folder_path) ):

        os.makedirs(average_map_folder_path)

    # store the average endpoint statistic map into a JSON file.
    with open(average_endpoint_statistic_map_file_path, 'w+') as json_file:

        json.dump(average_endpoint_statistic_map.tolist(), json_file)


def calculate_average_endpoint_statistic_maps(directory):
    '''

    Purpose:

        This function interates overll the folders in the given directory

        which pertain to a minimum required number of seizures in the baseline

        period, and calculates a set of endpoint statistics maps for each endpoint

        to be store in a 'final folder'. This function assumes that a text file called

        'meta_data.txt' which contains relevant information about the endpoint statistic
        
        maps is located in the same folder as this script.

    Inputs:

        1) directory:

            (string) - the name of the directory which contains the folders 
                       
                       in which the intermediate JSON files are stored

    Outputs:

        Technically None

    '''
    # set the array of endpoint statistic types, which is hardcoded into the software
    endpoint_statistic_map_file_names = ['expected_placebo_RR50_map', 'expected_placebo_MPC_map', 'expected_placebo_TTP_map',
                                         'expected_drug_RR50_map',    'expected_drug_MPC_map',    'expected_drug_TTP_map',
                                         'RR50_stat_power_map',       'MPC_stat_power_map',       'TTP_stat_power_map',
                                         'RR50_type_1_error_map',     'MPC_type_1_error_map',     'TTP_type_1_error_map']

    # set the file name of the meta-data text file, which is hardcoded into the software, along with its absolute file path
    meta_data_file_name = 'meta_data.txt'
    meta_data_file_path = directory + '/' + meta_data_file_name

    # read the relevant information from the meta-data text file
    with open( meta_data_file_path, 'r' ) as meta_data_text_file:

        # read information about the monthly seizure count mean axis
        start_monthly_mean = int(meta_data_text_file.readline())
        stop_monthly_mean  = int(meta_data_text_file.readline())
        step_monthly_mean  = int(meta_data_text_file.readline())

        # read information about the monthly seizure count standard deviation axis
        start_monthly_std_dev = int(meta_data_text_file.readline())
        stop_monthly_std_dev  = int(meta_data_text_file.readline())
        step_monthly_std_dev  = int(meta_data_text_file.readline())

        # read information about the maximum of the minimum required number of seizures in the baseline period
        max_min_req_base_sz_count = int(meta_data_text_file.readline())

        # read information about the number of maps to average over
        num_maps = int(meta_data_text_file.readline())

    # calculate the number of ticks on both axes
    num_monthly_means = int( (stop_monthly_mean - start_monthly_mean)/step_monthly_mean ) + 1
    num_monthly_std_devs = int( (stop_monthly_std_dev - start_monthly_std_dev)/step_monthly_std_dev ) + 1

    # iterate over all the minimum required numbers of seziures in the baseline period, from 0 all the way up to the maximum
    for min_req_base_sz_count in range(max_min_req_base_sz_count + 1):
    
        # iterate over all the endpoint statistic types
        for endpoint_statistic_type in endpoint_statistic_map_file_names:

            # calculate and store the avaerage endpoint statistic map for a given minimum required number of baseline seizures and a given endpoint statistic type
            calculate_average_endpoint_statistic_map(directory, min_req_base_sz_count, endpoint_statistic_type, 
                                                     num_monthly_means, num_monthly_std_devs,  num_maps)


def generate_model_patient_data(shape, scale, alpha, beta, num_patients_per_model, num_months_per_patient):
    '''
    
    Purpose:
    
        This function generates monthly seizure diaries which are all of equal length. The monthly seizure

        counts are generated by a negative binomial distribution with priors on both input parameters to the 
    
        negative binomial, resulting in four parameters as inputs to what is referred to as the NV model. The

        priors introduce heterogeneity between patients. 

    Inputs:

        1) shape:

            (float) - the first parameter for the NV model
                
        2) scale:

            (float) - the second parameter for the NV model
                
        3) alpha:

            (float) - the third parameter for the NV model
                
        4) beta:

            (float) - the fourth parameter for the NV model

        5) num_patients:

            int) - the number of seizure diaries to generate for this model
                
        6) num_months_per_patient:

            (int) - the number of months each seizure diary should have, this is the same across all patients

    Outputs:

        1) model_monthly_count_averages:

            (1D Numpy array) - the averages of each generated seizure diary
                
        2) model_standard_deviations:

            (1D Numpy array) - the standard deviations of each generated seizure diary
    
    '''
    
    # initialize array of monthly counts
    monthly_counts = np.zeros((num_patients_per_model, num_months_per_patient))
    
    # initialize arrays that will store the monthly seizure count averages and standard deviation for each patient's seizure diary
    model_monthly_count_averages = np.zeros(num_patients_per_model)
    model_monthly_count_standard_deviations = np.zeros(num_patients_per_model)

    # for each patient to be generated:
    for patient_index in range(num_patients_per_model):

        # set up a boolean flag to determine whether or not the patiet's data seems overdispersed
        overdispersed = False

        # while the currently generated patient has not yet been determined to be overdispersed:
        while(not overdispersed):

            # generate an n and p parameter for a new patient
            n = np.random.gamma(shape, 1/scale)
            p = np.random.beta(alpha, beta)

            # for each month in this new patient's diary
            for month_index in range(num_months_per_patient):

                # generate a monthly count as according to n and p parameters and gamma-poisson mixture model
                monthly_rate = np.random.gamma(28*n, (1 - p)/p)
                monthly_count = np.random.poisson(monthly_rate)

                # store the monthly_count
                monthly_counts[patient_index, month_index] = monthly_count

            # calculate the average and standard deviation of the monthly seizure counts for this patient
            monthly_count_average = np.mean(monthly_counts[patient_index, :])
            monthly_count_std_dev = np.std(monthly_counts[patient_index, :])

            # if the standard deviation is greater than the square root of the mean
            if( monthly_count_std_dev > np.sqrt(monthly_count_average) ):

                # then the patients seizure diary can be said to be overdispersed
                overdispersed = True
        
        # store the monthly seizure count average and standard deviation for this patient
        model_monthly_count_averages[patient_index] = monthly_count_average
        model_monthly_count_standard_deviations[patient_index] = monthly_count_std_dev

    return [model_monthly_count_averages, model_monthly_count_standard_deviations]


def generate_model_histograms(directory, num_patients_per_model, num_months_per_patient,
                              NV_model_1_group_parameters, NV_model_2_group_parameters):

    # get the group-level parameters for NV Model 1
    shape_1 = NV_model_1_group_parameters[0]
    scale_1 = NV_model_1_group_parameters[1]
    alpha_1 = NV_model_1_group_parameters[2]
    beta_1  = NV_model_1_group_parameters[3]

    # get the group-level parameters for NV Model 2
    shape_2 = NV_model_2_group_parameters[0]
    scale_2 = NV_model_2_group_parameters[1]
    alpha_2 = NV_model_2_group_parameters[2]
    beta_2  = NV_model_2_group_parameters[3]

    # set the file name of the meta-data text file, which is hardcoded into the software, along with its absolute file path
    meta_data_file_name = 'meta_data.txt'
    meta_data_file_path = directory + '/' + meta_data_file_name

    # read the relevant information from the meta-data text file
    with open( meta_data_file_path, 'r' ) as meta_data_text_file:

        # read information about the monthly seizure count mean axis
        start_monthly_mean = int( meta_data_text_file.readline() )
        stop_monthly_mean = int( meta_data_text_file.readline() )
        step_monthly_mean = int( meta_data_text_file.readline() )

        # read information about the monthly seizure count standard deviation axis
        start_monthly_std_dev = int( meta_data_text_file.readline() )
        stop_monthly_std_dev = int( meta_data_text_file.readline() )
        step_monthly_std_dev = int( meta_data_text_file.readline() )
        
        # read information about the maximum of the minimum required number of seizures in the baseline period
        max_min_req_base_sz_count = int( meta_data_text_file.readline() )

    # calculate what the length of the monthly mean and monthly standard deviation axes should be
    monthly_mean_axis_len = ( (stop_monthly_mean - start_monthly_mean)/step_monthly_mean ) + 1
    monthly_std_dev_axis_len = ( (stop_monthly_std_dev - start_monthly_std_dev)/step_monthly_std_dev ) + 1

    # generate Model 1 patients
    [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = \
                                        generate_model_patient_data(shape_1, scale_1, alpha_1, beta_1,
                                                                    num_patients_per_model, num_months_per_patient)
    
    # calculate the histogram of the Model 1 patients
    [H_model_1, _, _] = np.histogram2d(model_1_monthly_count_averages, model_1_monthly_count_standard_deviations, bins=[monthly_std_dev_axis_len, monthly_mean_axis_len], 
                                       range=[[start_monthly_mean, stop_monthly_mean], [start_monthly_std_dev, stop_monthly_std_dev]])
    H_model_1 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_1))))
    norm_const_1 = np.sum(np.sum(H_model_1, 0))
    H_model_1 = H_model_1/norm_const_1

    # generate Model 2 patients
    [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = \
                                    generate_model_patient_data(shape_2, scale_2, alpha_2, beta_2,
                                                                num_patients_per_model, num_months_per_patient)

    # calculate the histogram of the Model 2 patients
    [H_model_2, _, _] = np.histogram2d(model_2_monthly_count_averages, model_2_monthly_count_standard_deviations, bins=[monthly_std_dev_axis_len, monthly_mean_axis_len], 
                                       range=[[start_monthly_mean, stop_monthly_mean], [start_monthly_std_dev, stop_monthly_std_dev]])
    H_model_2 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_2))))
    norm_const_2 = np.sum(np.sum(H_model_2, 0))
    H_model_2 = H_model_2/norm_const_2

    return [H_model_1, H_model_2, max_min_req_base_sz_count]


def calculate_expected_model_responses(directory, num_patients_per_model, num_months_per_patient,
                                       NV_model_1_group_parameters, NV_model_2_group_parameters):

    endpoint_statistic_map_file_names = ['expected_placebo_RR50_map', 'expected_placebo_MPC_map', 'expected_placebo_TTP_map',
                                         'expected_drug_RR50_map',    'expected_drug_MPC_map',    'expected_drug_TTP_map',
                                         'RR50_stat_power_map',       'MPC_stat_power_map',       'TTP_stat_power_map',
                                         'RR50_type_1_error_map',     'MPC_type_1_error_map',     'TTP_type_1_error_map']
    
    NV_models_endpoint_statistics = [['Model 1 expected placebo RR50:  ', 'Model 2 expected placebo RR50:  '], 
                                     ['Model 1 expected placebo MPC:   ', 'Model 2 expected placebo MPC:   '], 
                                     ['Model 1 expected placebo TTP:   ', 'Model 2 expected placebo TTP:   '],
                                     ['Model 1 expected drug RR50:     ', 'Model 2 expected drug RR50:     '], 
                                     ['Model 1 expected drug MPC:      ', 'Model 2 expected drug MPC:      '], 
                                     ['Model 1 expected drug TTP:      ', 'Model 2 expected drug TTP:      '],
                                     ['Model 1 RR50 statistical power: ', 'Model 2 RR50 statistical power: '],
                                     ['Model 1 MPC statistical power:  ', 'Model 2 MPC statistical power:  '],
                                     ['Model 1 TTP statistical power:  ', 'Model 2 TTP statistical power:  '],
                                     ['Model 1 RR50 type-1 error:      ', 'Model 2 RR50 type-1 error:      '],
                                     ['Model 1 MPC type-1 error:       ', 'Model 2 MPC type-1 error:       '],
                                     ['Model 1 TTP type-1 error:       ', 'Model 2 TTP type-1 error:       ']]

    # generate the histogram for NV models 1 and 2, as well as obtaining the maximum value for the eligibility criteria
    [H_model_1, H_model_2, max_min_req_base_sz_count] = \
        generate_model_histograms(directory, num_patients_per_model, num_months_per_patient,
                                  NV_model_1_group_parameters, NV_model_2_group_parameters)

    # iterate over all the minimum required numbers of seizures in the baseline period, from 0 all the way up to the maximum
    for min_req_base_sz_count in range(max_min_req_base_sz_count + 1):
    
        # create both the folder path and the file path for the text file that will contain the NV model endpoint responses for this specific eligbility criteria
        NV_model_response_folder_path = directory + '/maps/NV_model_responses'
        NV_model_response_file_path = NV_model_response_folder_path + '/eligibility_criteria_' + str(min_req_base_sz_count) +'.txt'
        if ( not os.path.exists(NV_model_response_folder_path) ):
            os.makedirs(NV_model_response_folder_path)

        # iterate over all the endpoint statistic types
        for endpoint_statistic_type_index in range(len(endpoint_statistic_map_file_names)):
        
            # locate the json file containing the heatmap of the current endpoint statistic
            endpoint_statistic_type = endpoint_statistic_map_file_names[endpoint_statistic_type_index]
            endpoint_statistic_type_map_path = directory + '/' + str(min_req_base_sz_count) + '/final/' + endpoint_statistic_type + '.json'
            with open(endpoint_statistic_type_map_path, 'r') as json_file:
                endpoint_statistic_type_map = np.array(json.load(json_file))

            # calculate NV model endpoint statistics via 2D integration over the multiplication of the NV model histograms and endpoint response heatmaps
            Model_1_endpoint_statistic_response = np.sum(np.nansum(np.multiply(H_model_1, endpoint_statistic_type_map), 0))
            Model_2_endpoint_statistic_response = np.sum(np.nansum(np.multiply(H_model_2, endpoint_statistic_type_map), 0))

            # convert the NV model endpoint statistics into a presentable string
            Model_1_endpoint_statistic_response_str = NV_models_endpoint_statistics[endpoint_statistic_type_index][0] + str(np.round(100*Model_1_endpoint_statistic_response, 3))
            Model_2_endpoint_statistic_response_str = NV_models_endpoint_statistics[endpoint_statistic_type_index][1] + str(np.round(100*Model_2_endpoint_statistic_response, 3))
            Model_endpoint_statistic_responses_str = Model_1_endpoint_statistic_response_str + '\n' + Model_2_endpoint_statistic_response_str

            with open(NV_model_response_file_path, 'a+') as text_file:
                text_file.write(Model_endpoint_statistic_responses_str)


if (__name__=='__main__'):
    '''

        The purpose of this if statement is two-folder: 1) to act as the

        go-between for this python script and its wrapper shell script, and

        2) to act as the main point of coordination for this relatively

        uncomplicated python script.

    '''

    # take in the directory containing all the JSON files from the user
    directory = sys.argv[1]

    num_patients_per_model = int(sys.argv[2])
    num_months_per_patient = int(sys.argv[3])

    # get the group-level parameters for NV Model 1
    shape_1 = float(sys.argv[4])
    scale_1 = float(sys.argv[5])
    alpha_1 = float(sys.argv[6])
    beta_1  = float(sys.argv[7])
    NV_model_1_group_parameters = np.array([shape_1, scale_1, alpha_1, beta_1])

    # get the group-level parameters for NV Model 2
    shape_2 = float(sys.argv[8])
    scale_2 = float(sys.argv[9])
    alpha_2 = float(sys.argv[10])
    beta_2  = float(sys.argv[11])
    NV_model_2_group_parameters = np.array([shape_2, scale_2, alpha_2, beta_2])

    # caclulate the average endpoint statistic heatmaps
    calculate_average_endpoint_statistic_maps(directory)

    # calculate the predicted endpoint statistics of the NV models 
    calculate_expected_model_responses(directory, num_patients_per_model, num_months_per_patient,
                                       NV_model_1_group_parameters, NV_model_2_group_parameters)
    
