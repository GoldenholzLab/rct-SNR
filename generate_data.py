import numpy as np
import os
import json
import pandas as pd
import time

def generate_daily_seizure_diaries(daily_mean, daily_std_dev, num_patients, 
                                   num_baseline_days, num_testing_days, 
                                   min_req_base_sz_count):
    '''

    This function generates an array of equal-length daily seizure diaries, where the mean and standard deviation of the daily seizure 
    
    counts for each individual seizure diary are the same across all seizure diaries. Furthermore, the seizure diaries are designed to 
    
    be split up into a baseline and testing period, and are generated such that the number of baseline seizure counts each patient has 
    
    will not be below an integer specififed by the user, in order to enforce eligibility criteria. 


    Inputs:

        1) daily_mean:

            (float) - the mean of the daily seizure counts in each patient's seizure diary

        2) daily_std_dev:

            (float) - the standard deviation of the daily seizure counts in each patient's seizure diary

        3) num_patients:

            (int) - the number of seizure diaries in this array of seizure diaries
        
        4) num_baseline_days:

            (int) - the number of baseline days in each patient's seizure diary
        
        5) num_testing_days:

            (int) - the number of testing days in each patient's seizure diary
        
        6) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
    Outputs:

        1) daily_seizure_diaries:

            (2D Numpy array) - the array of daily seizure diaries in the form of a 2D Numpy array,
                               
                               the first dimension corresponds to the patient number while the

                               second dimension refers to the days of the seizure count being

                               accessed

    '''
    # calculate the total number of days from the number of baseline and testing intervals
    num_total_days = num_baseline_days + num_testing_days

    # initialize 2D array of daily seizure diaries for all patients
    daily_seizure_diaries = np.zeros((num_patients, num_total_days))

    # calculate the daily overdispersion parameter
    daily_mean_squared = np.power(daily_mean, 2)
    daily_var = np.power(daily_std_dev, 2)
    daily_overdispersion = (daily_var  - daily_mean)/daily_mean_squared

    # for each patient in the array of seizure diaries
    for patient_index in range(num_patients):

        # initialize a flag to decide whether or not this patient satisfies the eligibility
        # criteria of the minimum required baseline counts
        acceptable_baseline_counts = False

        # while the baseline rate is not acceptable
        while(not acceptable_baseline_counts):

            # for each interval in this patient's seizure diary
            for day_index in range(num_total_days):

                # generate daily seizure counts according to negative binomial distribution, as implemented by Gamma-Poisson mixture
                daily_rate = np.random.gamma(1/daily_overdispersion, daily_overdispersion*daily_mean)
                daily_count = np.random.poisson(daily_rate)

                # store each seizure count in each respective patient's seizure diary
                daily_seizure_diaries[patient_index, day_index] = daily_count
            
            # if the summation of all seizures in the baseline is greater than or equal to the minimum required number of baseline seizure counts
            if( np.sum( daily_seizure_diaries[patient_index, 0:num_baseline_days] ) >= min_req_base_sz_count ):

                # say that this patient satsifies the eligibility criteria
                acceptable_baseline_counts = True
    
    return daily_seizure_diaries


def calculate_percent_changes(daily_seizure_diaries, num_baseline_days, num_patients):
    '''

    This function calculates the percent change per patient given a set of equal-length daily

    seizure diaries.

    Inputs:

        1) daily_seizure_diaries:

            (2D Numpy array) - the array of daily seizure diaries in the form of a 2D Numpy array,
                               
                               the first dimension corresponds to the patient number while the

                               second dimension refers to the days of the seizure count being

                               accessed
        
        2) num_baseline_days:

            (int) - (int) - the number of baseline dayss in each patient's seizure diary

        3) num_patients:

            (int) - the number of seizure diaries in this array of seizure diaries

    Outputs:

        1) percent_changes:

            (1D Numpy array) - an array of percent changes, one percent change for each patient
                               
                               with a seizure diary

    '''

    # separate the daily seizure diaries into baseline and testing periods
    baseline_daily_seizure_diaries = daily_seizure_diaries[:, 0:num_baseline_days]
    testing_daily_seizure_diaries = daily_seizure_diaries[:, num_baseline_days:]

    # calculate the seizure frequencies for the baseline and testing period of each patient's seizure daily seizure diary
    baseline_daily_seizure_frequencies = np.mean(baseline_daily_seizure_diaries, 1)
    testing_daily_seizure_frequencies = np.mean(testing_daily_seizure_diaries, 1)

    # for each patient' seizure diary
    for patient_index in range(num_patients):

        # if the baseline seizure frequency is zero
        if(baseline_daily_seizure_frequencies[patient_index] == 0):

            # set it to a really small number to avoid divide-by-zero errors when calculating the percent changes
            baseline_daily_seizure_frequencies[patient_index] = 0.000000001

    # calculate the percent changes (percent change per patient)
    percent_changes = np.divide(baseline_daily_seizure_frequencies - testing_daily_seizure_frequencies, baseline_daily_seizure_frequencies)

    return percent_changes


def estimate_expected_endpoints(monthly_mean, monthly_std_dev, 
                                num_baseline_months, num_testing_months, 
                                min_req_base_sz_count, num_patients_per_trial, num_trials):
    '''

    This function estimates what the expected placebo response should be for a patient with a

    monthly mean and monthly standard deviation as specified by the input parameters. This function 

    will just return NaN if the standard deviation is less than the square root of the mean due to

    mathematical restrictions on the negative binomial distribution which is generating all these

    seizure counts. This function will also return NaN if the given monthly mean is just zero.

    Inputs:

        1) monthly_mean:

            (float) - the mean of the monthly seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the monthly seizure counts in each patient's seizure diary

        3) num_baseline_months:
        
            (int) - the number of baseline months in each patient's seizure diary

        4) num_testing_months:

            (int) - the number of testing months in each patient's seizure diary

        5) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts

        6) num_patients_per_trial:

            (int) - the number of patients generated per trial

        7) num_trials:

            (int) -  the number of trials used to estimate the expected endpoints
    
    Outputs:

        1) expected_RR50:

            (float) - the 50% responder rate which is expected from an individual with this specific 

                      monthly mean and monthly standard deviation
        
        1) expected_MPC:

            (float) - the median percent change which is expected from an individual with this specific 

                      monthly mean and monthly standard deviation

    '''

    # make sure that the patient does not have a true mean seizure count of 0
    if(monthly_mean != 0):

        # make sure that the patient is supposed to have overdispersed data
        if(monthly_std_dev > np.sqrt(monthly_mean)):

            # convert the monthly mean and monthly standard deviation into a daily mean and daily standard deviation
            daily_mean = monthly_mean/28
            daily_std_dev = monthly_std_dev/np.sqrt(28)

            # convert the the number of baseline months and testing months into baseline days and testing days
            num_baseline_days = num_baseline_months*28
            num_testing_days = num_testing_months*28

            # initialize the array that will contain the 50% responder rates and median percent changes from every trial
            RR50_array = np.zeros(num_trials)
            MPC_array = np.zeros(num_trials)

            # for every trial:
            for trial_index in range(num_trials):

                # generate one set of daily seizure diaries for each trial
                '''
                In the future, I will create two sets of diaries: one from placebo arm and one from drug arm.
                '''
                daily_seizure_diaries = \
                    generate_daily_seizure_diaries(daily_mean, daily_std_dev, num_patients_per_trial, 
                                                   num_baseline_days, num_testing_days, 
                                                   min_req_base_sz_count)

                # calculate the percent changes
                percent_changes = calculate_percent_changes(daily_seizure_diaries, num_baseline_days, num_patients_per_trial)

                # calculate the endpoints for this trial
                RR50 = 100*np.sum(percent_changes >= 0.5)/num_patients_per_trial
                MPC = 100*np.median(percent_changes)

                # store the endpoints in their respective arrays
                RR50_array[trial_index] = RR50
                MPC_array[trial_index] = MPC

            # calculate the means of the endpoints over all trials
            expected_RR50 = np.mean(RR50_array)
            expected_MPC = np.mean(MPC_array)

            return [expected_RR50, expected_MPC]
    
        # if the patient does not have overdispersed data:
        else:

            # say that calculating their placebo response is impossible
            return [np.nan, np.nan]

    # if the patient does not have a true mean seizure count of 0, then:
    else:

        # say that calculating their placebo response is impossible
        return [np.nan, np.nan]


def generate_expected_endpoint_maps(start_monthly_mean,       stop_monthly_mean,    step_monthly_mean, 
                                    start_monthly_std_dev,    stop_monthly_std_dev, step_monthly_std_dev,
                                    num_baseline_months,      num_testing_months,   min_req_base_sz_count, 
                                    num_patients_per_trial,   num_trials):

    # create the monthly mean and monthly standard deviation axes
    monthly_mean_array = np.arange(start_monthly_mean, stop_monthly_mean + step_monthly_mean, step_monthly_mean)
    monthly_std_dev_array = np.arange(start_monthly_std_dev, stop_monthly_std_dev + step_monthly_std_dev, step_monthly_std_dev)

    # flip the monthly seizure count standard deviation axes
    monthly_std_dev_array = np.flip(monthly_std_dev_array, 0)

    # count up the number of locations on both axes
    num_monthly_means = len(monthly_mean_array)
    num_monthly_std_devs = len(monthly_std_dev_array)

    # initialize the 2D numpy arrays that will hold the expected endpoint maps
    expected_RR50_endpoint_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_MPC_endpoint_map = np.zeros((num_monthly_std_devs, num_monthly_means))

    # for every given monthly standard deviation on the y-axis
    for monthly_std_dev_index in range(num_monthly_std_devs):

        # for every given monthly mean on the x-axis
        for monthly_mean_index in range(num_monthly_means):

            # access the corresponding actual monthly mean and monthly standard deviation
            monthly_mean = monthly_mean_array[monthly_mean_index]
            monthly_std_dev = monthly_std_dev_array[monthly_std_dev_index]

            # start keeping track of how long it takes to estimate the expected endpoints for the given mean and standard deviation
            start_time_in_seconds = time.time()

            # estimate the estimate the expected endpoints for the given mean and standard deviation
            [expected_RR50, expected_MPC] =  \
                estimate_expected_endpoints(monthly_mean, monthly_std_dev, 
                                            num_baseline_months, num_testing_months, 
                                            min_req_base_sz_count, num_patients_per_trial, num_trials)
            
            # put a stop to the timer on the endpoint estimation
            stop_time_in_seconds = time.time()
            
            # calculate the total number of minutes it took to estimate the expected endpoints for the given mean and standard deviation
            total_time_in_minutes = (stop_time_in_seconds - start_time_in_seconds)/60

            # store the esimtated expected endpoints
            expected_RR50_endpoint_map[monthly_std_dev_index, monthly_mean_index] = expected_RR50
            expected_MPC_endpoint_map[monthly_std_dev_index, monthly_mean_index] = expected_MPC

            # prepare a string telling the user where the algorithm is in terms of map generation
            cpu_time_string = 'cpu time (minutes): ' + str(np.round(total_time_in_minutes, 2))
            RR50_string = 'expected RR50: ' +  str(np.round(expected_RR50, 2))
            MPC_string = 'expected MPC: ' +  str(np.round(expected_MPC, 2))
            monthly_mean_string = str(np.round(monthly_mean, 2))
            monthly_std_dev_string = str(np.round(monthly_std_dev, 2))
            orientation_string = 'monthly mean, monthly standard deviation: (' + monthly_mean_string + ', ' + monthly_std_dev_string + ')'
            data_string = '\n\n' + orientation_string + ':\n' + RR50_string + '\n' + MPC_string + '\n' + cpu_time_string

            # print the string
            print(data_string)
    
    return [expected_RR50_endpoint_map, expected_MPC_endpoint_map]


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


start_monthly_mean = 0
stop_monthly_mean = 16
step_monthly_mean = 1

start_monthly_std_dev = 0
stop_monthly_std_dev = 16
step_monthly_std_dev = 1

monthly_mean = 8.7
monthly_std_dev = 2.95
num_baseline_months = 2
num_testing_months = 3
min_req_base_sz_count = 4
num_patients_per_trial = 153
num_trials = 10

[expected_RR50_endpoint_map, expected_MPC_endpoint_map] = \
    generate_expected_endpoint_maps(start_monthly_mean,       stop_monthly_mean,    step_monthly_mean, 
                                    start_monthly_std_dev,    stop_monthly_std_dev, step_monthly_std_dev,
                                    num_baseline_months,      num_testing_months,   min_req_base_sz_count, 
                                    num_patients_per_trial,   num_trials)







'''
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
'''