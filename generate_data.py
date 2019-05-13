import numpy as np
import os
import json

def generate_patients(mean, std_dev, time_scale_const, num_patients, num_baseline_intervals, num_testing_intervals, min_req_base_sz_count):

    # calculate the total number of intervals from the number of baseline and testing intervals
    num_total_intervals = num_baseline_intervals + num_testing_intervals

    # initialize 2D array of seizure diaries for all patients
    seizure_diaries = np.zeros((num_patients, num_total_intervals))

    # calculate the overdispersion parameter
    mean_squared = np.power(mean, 2)
    var = np.power(std_dev, 2)
    overdispersion = (var  - mean)/mean_squared

    # for each patient in the array of seizure diaries
    for patient_index in range(num_patients):

        # initialize a flag to decide whether or not this patient satisfies the eligibility
        # criteria of the minimum required baseline counts
        acceptable_baseline_counts = False

        # while the baseline rate is not acceptable
        while(not acceptable_baseline_counts):

            for interval_index in range(num_total_intervals):

                rate = np.random.gamma(time_scale_const/overdispersion, overdispersion*mean)
                count = np.random.poisson(rate)

                seizure_diaries[patient_index, interval_index] = count
            
            if( np.sum( seizure_diaries[patient_index, 0:num_baseline_intervals] ) >= min_req_base_sz_count ):

                acceptable_baseline_counts = True
    
    return seizure_diaries
        


def generate_map(start_monthly_mu,    stop_monthly_mu,    step_monthly_mu, 
                 start_monthly_sigma, stop_monthly_sigma, step_monthly_sigma):

    monthly_mu_array = np.arange(start_monthly_mu, stop_monthly_mu + step_monthly_mu, step_monthly_mu)
    monthly_sigma_array = np.arange(start_monthly_sigma, stop_monthly_sigma + step_monthly_sigma, step_monthly_sigma)

    monthly_sigma_array = np.flip(monthly_sigma_array, 0)

    num_monthly_mus = len(monthly_mu_array)
    num_monthly_sigmas = len(monthly_sigma_array)

    data_map = np.zeros((num_monthly_sigmas, num_monthly_mus))

    for monthly_sigma_index in range(num_monthly_sigmas):

        for monthly_mu_index in range(num_monthly_mus):

            monthly_mu = monthly_mu_array[monthly_mu_index]
            monthly_sigma = monthly_sigma_array[monthly_sigma_index]

            if(monthly_sigma > np.sqrt(monthly_mu)):

                data_map[monthly_sigma_index, monthly_mu_index] = monthly_mu + monthly_sigma
            
            else:
                
                data_map[monthly_sigma_index, monthly_mu_index] = np.nan
    
    return data_map


def store_map(data_map, 
              data_map_file_name, data_map_meta_data_file_name,
              x_axis_start, x_axis_stop, x_axis_step,
              y_axis_start, y_axis_stop, y_axis_step):

    data_map_file_path = os.getcwd() + '/' + data_map_file_name + '.json'

    data_map_metadata_file_path = os.getcwd() + '/' + data_map_meta_data_file_name + '.json'

    metadata = np.array([x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step])

    with open(data_map_file_path, 'w+') as map_storage_file:

        json.dump(data_map.tolist(), map_storage_file)

    with open(data_map_metadata_file_path, 'w+') as map_metadata_storage_file:

        json.dump(metadata.tolist(), map_metadata_storage_file)


start_monthly_mu = 0
stop_monthly_mu = 16
step_monthly_mu = 1

start_monthly_sigma = 0
stop_monthly_sigma = 16
step_monthly_sigma = 1

test_map_file_name = 'test_map'
test_map_metadata_file_name = 'test_map_metadata'

test_map = generate_map(start_monthly_mu,    stop_monthly_mu,    step_monthly_mu, 
                        start_monthly_sigma, stop_monthly_sigma, step_monthly_sigma)

store_map(test_map, 
          test_map_file_name, test_map_metadata_file_name,
          start_monthly_mu,    stop_monthly_mu,    step_monthly_mu,
          start_monthly_sigma, stop_monthly_sigma, step_monthly_sigma)

