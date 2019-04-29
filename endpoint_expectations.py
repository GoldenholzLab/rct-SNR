import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import time
import psutil


def estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin, num_days_per_month, 
                                num_months_per_patient_baseline, num_months_per_patient_testing):
    '''

    Inputs:

        1) monthly_mu:

            (float) - 

        2) monthly_sigma:

            (float) - 
        
        3) num_patients_per_trial:

            (int) - 
        
        4) num_trials_per_bin:

            (int) - 
        
        5) num_days_per_month:

            (int) - 
        
        6) num_months_per_patient_baseline:

            (int) - 
        
        7) num_months_per_patient_testing:

            (int) - 

    Outputs:

        1) expected_RR50:

            (float) - 
        
        2) expected_MPC:

            (float) - 

    '''

    # calculate the daily mean and daily standard deviation of the counts, as well as the total number of months
    daily_mu = monthly_mu/num_days_per_month
    daily_sigma = monthly_sigma/np.sqrt(num_days_per_month)
    num_months_per_patient_total = num_months_per_patient_baseline + num_months_per_patient_testing

    # check to see if the mean is not zero
    daily_mu_not_zero = daily_mu != 0

    # if the mean is not zero...
    if( daily_mu_not_zero ):

        # check to see if the daily standard deviation is greater than the square root of the daily mean
        daily_sigma_greater_than_square_root_of_daily_mu = daily_sigma > np.sqrt(daily_mu)

        # if the daily standard deviation is greater than the square root of the daily mean...
        if( daily_sigma_greater_than_square_root_of_daily_mu ):

            # calculate the overdispersion parameter
            daily_mu_squared = np.power(daily_mu, 2)
            daily_var = np.power(daily_sigma, 2)
            daily_alpha = (daily_var - daily_mu)/daily_mu_squared

            # initialize the arrays that will store the RR50 and MPC from each trial
            RR50_array = np.zeros(num_trials_per_bin)
            MPC_array = np.zeros(num_trials_per_bin)

            # for every trial...
            for trial_index in range(num_trials_per_bin):

                # initialize the 2D array of monthly counts from multiple patients over one trial
                monthly_counts = np.zeros((num_patients_per_trial, num_months_per_patient_total))

                # for each individual patient
                for patient_index in range(num_patients_per_trial):

                    # for each month in each individual patient's diary
                    for month_index in range(num_months_per_patient_total):

                        # generate a monthly count according to gamma-poisson mixture
                        monthly_rate = np.random.gamma(num_days_per_month/daily_alpha, daily_alpha*daily_mu)
                        monthly_count = np.random.poisson(monthly_rate)

                        # store the monthly count
                        monthly_counts[patient_index, month_index] = monthly_count 

                # separate the monthly counts into baseline and testing periods
                baseline_monthly_counts = monthly_counts[:, 0:num_months_per_patient_baseline]
                testing_monthly_counts = monthly_counts[:, num_months_per_patient_baseline:]
        
                # calculate the seizure frequencies of the basline and test period for each patient
                baseline_monthly_frequencies = np.mean(baseline_monthly_counts, 1)
                testing_monthly_frequencies = np.mean(testing_monthly_counts, 1)

                # for each patient
                for patient_index in range(num_patients_per_trial):

                    # if the baseline seizure frequency turns out to zero...
                    if( baseline_monthly_frequencies[patient_index] == 0 ):

                        # make the baseline frequency into a very small positive number
                        baseline_monthly_frequencies[patient_index] = 0.000000000000000001
        
                # calculate the percent changes of the seizure frequencies across the baseline and testing periods for each patient
                percent_changes = np.divide(baseline_monthly_frequencies - testing_monthly_frequencies, baseline_monthly_frequencies)

                # calculate the RR50 and MPC for this trial
                RR50 = 100*np.sum(percent_changes >= 0.5)/num_patients_per_trial
                MPC = 100*np.median(percent_changes)

                # store the RR50 and MPC
                RR50_array[trial_index] = RR50
                MPC_array[trial_index] = MPC
        
            # calculate the mean RR50 and mean MPC over all the trials
            expected_RR50 = np.mean(RR50_array)
            expected_MPC = np.mean(MPC_array)

            return [expected_RR50, expected_MPC]
        
        else:

            # if the square root of the daily mean is less than the daily standard deviation, then the data is not underdispersed
            # therefore, it does not make sense to model this mean-and-standard-deviation pair
            return [np.NaN, np.NaN]
    
    else:

        # if the daily mean is zero, then it also does not make sense to model that pair
        return [np.NaN, np.NaN]


def generate_expected_endpoint_maps(monthly_mu_start, monthly_mu_stop, monthly_mu_step, monthly_sigma_start, monthly_sigma_stop, monthly_sigma_step,
                                    num_patients_per_trial, num_trials_per_bin, num_days_per_month, num_months_per_patient_baseline, num_months_per_patient_testing):
    '''

    Inputs:

        1) monthly_mu_start

            (float) - 

        2) monthly_mu_stop

            (float) - 
        
        3) monthly_mu_step:

            (float) - 
        
        4) monthly_sigma_start:

            (float) - 

        5) monthly_sigma_stop:

            (float) - 
        
        6) monthly_sigma_step:

            (float) - 
        
        7) num_patients_per_trial:

            (int) - 
        
        8) num_trials_per_bin:

            (int) - 
        
        9) num_days_per_month:
            
            (int) - 
        
        10) num_months_per_patient_baseline:

            (int) - 
        
        11) num_months_per_patient_testing:

            (int) - 

    Outputs:

        1) expected_RR50_map:

            (2D Numpy array) - 
        
        2) expected_MPC_array:

            (2D Numpy array) - 

    '''

    # initialize the axes of the maps to be generated
    monthly_mu_array = np.arange(monthly_mu_start, monthly_mu_stop + monthly_mu_step, monthly_mu_step)
    monthly_sigma_array = np.arange(monthly_sigma_start, monthly_sigma_stop + monthly_sigma_step, monthly_sigma_step)

    # get the lengths of each axis
    num_monthly_mus = len(monthly_mu_array)
    num_monthly_sigmas = len(monthly_sigma_array)

    # initialize the 2D arrays which will become the expected endpoint value maps 
    expected_RR50_map = np.zeros((num_monthly_sigmas, num_monthly_mus))
    expected_MPC_map = np.zeros((num_monthly_sigmas, num_monthly_mus))

    # for every monthly mean
    for monthly_mu_index in range(num_monthly_mus):

        # for every monthly standard deviation
        for monthly_sigma_index in range(num_monthly_sigmas):

            # extract the monthly mean and standard deviations for on particular location
            monthly_mu = monthly_mu_array[monthly_mu_index]
            monthly_sigma = monthly_sigma_array[monthly_sigma_index]

            # calculate the exepcted RR50 and the expected MPC for that location
            [expected_RR50, expected_MPC] = estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin, num_days_per_month, 
                                                                        num_months_per_patient_baseline, num_months_per_patient_testing)

            # palce the respective endpoint calculations into the relevant location in their own maps
            expected_RR50_map[monthly_sigma_index, monthly_mu_index] = expected_RR50
            expected_MPC_map[monthly_sigma_index, monthly_mu_index] = expected_MPC

            print('monthly mu: ' + str(np.round(monthly_mu, 2)) + '\nmonthly_sigma: ' + str(np.round(monthly_sigma, 2)) + '\nexpected RR50: ' + str(np.round(expected_RR50, 2)) + '\nexpected MPC: ' + str(np.round(expected_MPC, 2)) + '\n\n')

    # format the maps to be more readable
    expected_RR50_map = np.flipud(expected_RR50_map)
    expected_MPC_map = np.flipud(expected_MPC_map)

    return [expected_RR50_map, expected_MPC_map]

def generate_model_patient_data(shape, scale, alpha, beta, num_patients_per_model, num_months_per_patient, num_days_per_month):
    '''

    Inputs:

        1) shape:

            (float) -
                
        2) scale:

            (float) - 
                
        3) alpha:

            (float) - 
                
        4) beta:

            (float) - 

        5) num_patients:

            int) - 
                
        6) num_months_per_patient:

            (int) - 
                
        7) num_days_per_month

            (int) - 

    Outputs:

        1) model_monthly_count_averages:

            (1D Numpy array) - 
                
        2) model_standard_deviations:

            (1D Numpy array) - 

    '''
        
    # initialize array of monthly counts
    monthly_counts = np.zeros((num_patients_per_model, num_months_per_patient))
        
    # over each patient
    for patient_index in range(num_patients_per_model):

        # generate an n and p parameter for each patient
        n = np.random.gamma(shape, 1/scale)
        p = np.random.beta(alpha, beta)

        # for each month in each patient's diary
        for month_index in range(num_months_per_patient):

            # generate a monthly count as according to n and parameters and gamma-poisson mixture model
            monthly_rate = np.random.gamma(num_days_per_month*n, (1 - p)/p)
            monthly_count = np.random.poisson(monthly_rate)

            # store monthly_count
            monthly_counts[patient_index, month_index] = monthly_count
        
    # calculate the average and standard deviation of the monthly seizure counts for each patient
    model_monthly_count_averages = np.mean(monthly_counts, 1)
    model_monthly_count_standard_deviations = np.std(monthly_counts, 1)

    return [model_monthly_count_averages, model_monthly_count_standard_deviations]

def generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, num_patients_per_model, num_months_per_patient, num_days_per_month,
                        monthly_mu_start, monthly_mu_stop, monthly_mu_step, monthly_sigma_start, monthly_sigma_stop, monthly_sigma_step,
                        num_patients_per_trial, num_trials_per_bin, num_months_per_patient_baseline, num_months_per_patient_testing):
    '''

    Inputs:

        1) shape_1:

            (float) - 
        
        2) scale_1:

            (float) - 
        
        3) alpha_1:
        
            (float) - 
        
        4) beta_1:
        
            (float) - 
            
        5) shape_2:

            (float) - 
        
        6) scale_2: 

            (float) - 

        7) alpha_2:

            (float) - 
         
        8) beta_2:
        
            (float) - 
        
        9) num_patients_per_model
        
            (int) - 
        
        10) num_months_per_patient
        
            (int) - 
        
        11) num_days_per_month:

            (int) - 
        
        12) monthly_mu_start:
        
            (float) -     
        
        13) monthly_mu_stop:
        
            (float) - 
        
        14) monthly_mu_step:
        
            (float) - 
        
        15) monthly_sigma_start:
        
            (float) - 
        
        16) monthly_sigma_stop:
        
            (float) - 
        
        17) monthly_sigma_step:

            (float) - 
        
        18) num_patients_per_trial:
        
            (int) - 
        
        19) num_trials_per_bin:
        
            (int) - 
        
        20) num_months_per_patient_baseline:
        
            (int) - 
        
        21) num_months_per_patient_testing:

            (int) - 

    Outputs:

        1) Model_1_expected_RR50:
        
            (float) - 
        
        2) Model_1_expected_MPC:
        
            (float) - 
        
        3) Model_2_expected_RR50:
        
            (float) - 
        
        4) Model_2_expected_MPC:
            
            (float) - 
        
        5) H_model_1:
        
            (2D Numpy Array) - 
        
        6) H_model_2:

            (2D NUmpy Array) - 

    '''
    # generate Model 1 patients
    [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = \
                                        generate_model_patient_data(shape_1, scale_1, alpha_1, beta_1,
                                                                num_patients_per_model, num_months_per_patient, num_days_per_month)

    # generate Model 2 patients
    [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = \
                                        generate_model_patient_data(shape_2, scale_2, alpha_2, beta_2, 
                                                                num_patients_per_model, num_months_per_patient, num_days_per_month)

    # generate the expected RR50 and expected MPC maps
    [expected_RR50_map, expected_MPC_map] = generate_expected_endpoint_maps(monthly_mu_start, monthly_mu_stop, monthly_mu_step, monthly_sigma_start, monthly_sigma_stop, monthly_sigma_step,
                                                                            num_patients_per_trial, num_trials_per_bin, num_days_per_month, num_months_per_patient_baseline, num_months_per_patient_testing)

    # get the size of those expected endpoint maps
    [nx, ny] = expected_RR50_map.shape

    # calculate the histogram of the Model 1 patients
    [H_model_1, _, _] = np.histogram2d(model_1_monthly_count_averages, model_1_monthly_count_standard_deviations, bins=[ny, nx], 
                                        range=[[monthly_mu_start, monthly_mu_stop], [monthly_sigma_start, monthly_sigma_stop]])
    H_model_1 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_1))))
    norm_const_1 = np.sum(np.sum(H_model_1, 0))
    H_model_1 = H_model_1/norm_const_1

    # calculate the histogram of the Model 2 patients
    [H_model_2, _, _] = np.histogram2d(model_2_monthly_count_averages, model_2_monthly_count_standard_deviations, bins=[ny, nx], 
                                        range=[[monthly_mu_start, monthly_mu_stop], [monthly_sigma_start, monthly_sigma_stop]])
    H_model_2 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_2))))
    norm_const_2 = np.sum(np.sum(H_model_2, 0))
    H_model_2 = H_model_2/norm_const_2

    # combine the expected endpoint maps as well as the Model 1 and Model 2 histograms to get the expected RR50 and MPC for both models
    Model_1_expected_RR50 = np.sum(np.nansum(np.multiply(H_model_1, expected_RR50_map), 0))
    Model_1_expected_MPC  = np.sum(np.nansum(np.multiply(H_model_1, expected_MPC_map),  0))
    Model_2_expected_RR50 = np.sum(np.nansum(np.multiply(H_model_2, expected_RR50_map), 0))
    Model_2_expected_MPC  = np.sum(np.nansum(np.multiply(H_model_2, expected_MPC_map),  0))

    return [Model_1_expected_RR50, Model_1_expected_MPC, Model_2_expected_RR50, Model_2_expected_MPC, expected_RR50_map, expected_MPC_map, H_model_1, H_model_2]


def create_plots(expected_RR50_map, expected_MPC_map, H_model_1, H_model_2, 
                 monthly_mu_start, monthly_mu_stop, monthly_sigma_start, monthly_sigma_stop):
    '''

    Inputs:

        1) expected_RR50_map:
            
            (2D Numpy Array) -
        
        2) expected_MPC_map: 
        
            (2D Numpy Array) - 
        
        3) H_model_1:
        
            (2D Numpy Array) - 
        
        4) H_model_2:
            
            (2D Numpy Array) - 
        
        5) monthly_mu_start:
        
            (float) - 
        
        6) monthly_mu_stop:
            
            (float) -  
        
        7) monthly_mu_step:

            (float) - 
        
        8) monthly_sigma_start:
        
            (float) - 
        
        9) monthly_sigma_stop:

            (float) - 
        
        10) monthly_sigma_step:

            (float) - 
        
        11) power_law_slopes:

            (1D Numpy Array) - 

    Outputs:

        Technically none, but practically, the output is four pyplot figures that are saved in the same folder as this script.

    '''

    mu_ticks = np.arange(monthly_mu_start, 1 + monthly_mu_stop, 1)
    sigma_ticks = np.flip( np.arange(monthly_sigma_start, 1 + monthly_sigma_stop, 1), 0 )

    fig1 = plt.figure()
    ax1 = sns.heatmap(expected_RR50_map)
    ax1.set_xticklabels( mu_ticks )
    ax1.set_yticklabels( sigma_ticks, rotation='horizontal' )
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected RR50 Placebo')
    #fig1.savefig('Expected RR50 map.png')

    fig2 = plt.figure()
    ax2 = sns.heatmap(expected_MPC_map)
    ax2.set_xticklabels( mu_ticks )
    ax2.set_yticklabels( sigma_ticks, rotation='horizontal' )
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected MPC Placebo')
    #fig2.savefig('Expected MPC map.png')

    fig3 = plt.figure()
    ax3 = sns.heatmap(H_model_1)
    ax3.set_xticklabels( mu_ticks )
    ax3.set_yticklabels( sigma_ticks, rotation='horizontal' )
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Model 1 patient population')
    #fig3.savefig('Model 1 patients 2D Histogram.png')

    fig4 = plt.figure()
    ax4 = sns.heatmap(H_model_2)
    ax4.set_xticklabels( mu_ticks )
    ax4.set_yticklabels( sigma_ticks, rotation='horizontal' )
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Model 2 patient population')
    #fig4.savefig('Model 2 patients 2D Histogram.png')


if (__name__ == '__main__'):

    start_time_in_seconds = time.time()

    # take in the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('array', nargs='+')
    args = parser.parse_args()
    arg_array = args.array

    # parameters for defining the x-axis of the maps which correspond to the mean
    monthly_mu_start = int(arg_array[0])
    monthly_mu_stop = int(arg_array[1])
    monthly_mu_step = float(arg_array[2])

    # parameters for defining the x-axis of the maps which correspond to the standard deviation
    monthly_sigma_start = int(arg_array[3])
    monthly_sigma_stop = int(arg_array[4])
    monthly_sigma_step = float(arg_array[5])

    # parameters for creating the expected endpoint response maps
    num_patients_per_trial = int(arg_array[6])
    num_trials_per_bin = int(arg_array[7])
    num_days_per_month = int(arg_array[8])
    num_months_per_patient_baseline = int(arg_array[9])
    num_months_per_patient_testing = int(arg_array[10])

    # parameters for generating the synthetic model 1 and model 2 patients
    num_patients_per_model = int(arg_array[11])
    num_months_per_patient = int(arg_array[12])

    # parameters for formatting the output
    model_expected_endpoints_file_name = arg_array[13]
    expected_endpoints_decimal_round = int(arg_array[14])
    plots_flag = bool(arg_array[15])

    '''
    monthly_mu_start = 0
    monthly_mu_stop = 15
    monthly_mu_step = 1

    monthly_sigma_start = 0
    monthly_sigma_stop = 15
    monthly_sigma_step = 1

    num_patients_per_trial = 150
    num_trials_per_bin = 30
    num_days_per_month = 28
    num_months_per_patient_baseline = 2
    num_months_per_patient_testing = 3

    num_patients_per_model = 10000
    num_months_per_patient = 24
    num_days_per_month = 28

    model_expected_endpoints_file_name = 'model_1_and_model_2_expected_endpoints'
    expected_endpoints_decimal_round = 3
    plots_flag = True
    '''

    shape_1 = 24.143
    scale_1 = 297.366
    alpha_1 = 284.024
    beta_1 = 369.628

    shape_2 = 111.313
    scale_2 = 296.728
    alpha_2 = 296.339
    beta_2 = 243.719

    folder = os.getcwd()
    model_expected_endpoints_file_path = folder + '/' + model_expected_endpoints_file_name + '.txt'

    [Model_1_expected_RR50, Model_1_expected_MPC, Model_2_expected_RR50, Model_2_expected_MPC, expected_RR50_map, expected_MPC_map, H_model_1, H_model_2] = \
                generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, num_patients_per_model, num_months_per_patient, num_days_per_month,
                                  monthly_mu_start, monthly_mu_stop, monthly_mu_step, monthly_sigma_start, monthly_sigma_stop, monthly_sigma_step,
                                  num_patients_per_trial, num_trials_per_bin, num_months_per_patient_baseline, num_months_per_patient_testing)


    text_data =     'Model 1 expected RR50: ' + str(np.round(Model_1_expected_RR50, expected_endpoints_decimal_round)) + \
                '\n\nModel 1 expected MPC: '  + str(np.round(Model_1_expected_MPC,  expected_endpoints_decimal_round)) + \
                '\n\nModel 2 expected RR50: ' + str(np.round(Model_2_expected_RR50, expected_endpoints_decimal_round)) + \
                '\n\nModel 2 expected MPC: '  + str(np.round(Model_2_expected_MPC,  expected_endpoints_decimal_round))                                  

    with open(model_expected_endpoints_file_path, 'w+') as model_expected_endpoint_text_file:

            model_expected_endpoint_text_file.write(text_data)

    if(plots_flag):

        create_plots(expected_RR50_map, expected_MPC_map, H_model_1, H_model_2, 
                     monthly_mu_start, monthly_mu_stop, monthly_sigma_start, monthly_sigma_stop)

    stop_time_in_seconds = time.time()
    total_time_in_seconds = stop_time_in_seconds - start_time_in_seconds
    total_time_in_minutes = total_time_in_seconds/60

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)

    print('\n\ntotal time in minutes: ' + str(total_time_in_minutes) + 
           '\nmemory used in GB: ' + str(used_mem_in_gigabytes) + '\n\n')
