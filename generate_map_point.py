import numpy as np
from scipy import stats
from lifelines.statistics import logrank_test
import os
import json
import sys
import time
import psutil


def generate_daily_seizure_diaries(daily_mean, daily_std_dev, num_patients, 
                                   num_baseline_days, num_testing_days, 
                                   min_req_base_sz_count):
    '''

    Purpose:

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


def apply_effect(effect_mu, effect_sigma, daily_seizure_diaries,
                 num_patients, num_baseline_days, num_testing_days):
    '''

    Purpose:

        This function modifies the seziure counts in the testing period of one patient's daily seizure diary 

        according to a randomly generated effect size (generated via Normal distribution). If the effect is

        postive, it removes seizures. If it is negative, it adds them.

    Inputs:

        1) effect_mu:

            (float) - the mean of the desired effect's distribution
        
        2) effect_sigma:

            (float) - the standard deviation of the desired effect's distribution

        3) daily_seizure_diaries:
        
            (2D Numpy array) - the seizure diaries of multiple patients, with each one being of equal length
                               
                               in both the baseline and testing period

        4) num_baseline_days:
        
            (int) - the number of days in the baseline period of each patients

        5) num_testing_days:

            (int) - the number of days in the testing period of each patient
        
    Outputs:
    
        1) daily_seizure_diary:

            (1D Numpy array) - the seizure diary of one patient, with seizure counts of the testing period modified by the effect

    '''

    # for each patient's seizure diary:
    for patient_index in range(num_patients):

        # randomly generate an effect from the given distribution
        effect = np.random.normal(effect_mu, effect_sigma)
    
        # if the effect is greater than 100%:
        if(effect > 1):
        
            # threshold it so that it cannot be greater than 100%
            effect = 1
    
        # extract only the seizure counts from the testing period of the patient's seizure diary
        testing_counts = daily_seizure_diaries[patient_index, num_baseline_days:]
    
        # for each seizure count in the testing period:
        for testing_count_index in range(num_testing_days):
        
            # initialize the number of seizures removed per each seizure count
            num_removed = 0

            # extract only the relevant seizure count from the testing period
            testing_count = testing_counts[testing_count_index]
            
            # for each individual seizure in the relevant seizure count
            for count_index in range(np.int_(testing_count)):
                
                # if a randomly generated number between 0 and 1 is less than or equal to the absolute value of the effect
                if(np.random.random() <= np.abs(effect)):
                    
                    # iterate the number of seizures removed by the sign of the effect
                    num_removed = num_removed + np.sign(effect)
        
            # remove (or add) the number of seizures from the relevant seizure count as determined by the probabilistic algorithm above
            testing_counts[testing_count_index] = testing_count  - num_removed
    
        # set the patient's seizure counts from the testing period equal to the newly modified seizure counts
        daily_seizure_diaries[patient_index, num_baseline_days:] = testing_counts
    
    return daily_seizure_diaries


def calculate_percent_changes(daily_seizure_diaries, num_baseline_days, num_patients):
    '''

    Purpose:

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


def calculate_times_to_prerandomization(daily_seizure_diaries, num_months_baseline, num_testing_days, num_patients):
    '''

    Purpose:

        This function calculates the time-to-prerandomization endpoint for each patient's daily seizure diary.

    Inputs:

        1) daily_seizure_diaries:
        
            (2D Numpy array) - an array of all the seizure diaries for each patient, all of equal length
                               
                               and with daily seizure counts inside

        2) num_months_baseline:

            (int) - the length of the baseline period in months
        
        3) num_testing_days:
        
            (int) - the lenght of the testing period in days
        
        4) num_patients:

            (int) - the number of seizure diaries (with each patient having one seizure diary, making this
                    
                    the same number as the number of patients)

    Outputs:

        1) ttp_times:

            (1D Numpy array) - an array of times-to-prerandomization, with one time for each patient's diary

    '''

    # calculate the number of baseline days
    num_days_in_one_month = 28
    num_baseline_days = num_months_baseline*num_days_in_one_month

    # separate the baseline period and testing period in each seizure diary
    baseline_daily_seizure_diaries = daily_seizure_diaries[:, :num_baseline_days]
    testing_daily_seizure_diaries = daily_seizure_diaries[:, num_baseline_days:]

    # initialize the 2D Numpy array which will hold all the monthly seizure diaries for the baseline period of each patient
    baseline_monthly_seizure_diaries = np.zeros((num_patients, num_months_baseline))

    # for each month in the baseline period:
    for month_index in range(num_months_baseline):
    
        # sort out which days in the seizure diary are the beginnning day and end day for each separate month
        beginning_day_index = 0 + num_days_in_one_month*month_index
        end_day_index = num_days_in_one_month*(1 + month_index)
    
        # extract the month in question using the daily indices found above
        month_of_baseline_seizure_counts = baseline_daily_seizure_diaries[:, beginning_day_index:end_day_index]
    
        # store the monthly seizure counts as the sum of the daily seizure counts over 28 days for each month in each diary
        baseline_monthly_seizure_diaries[:, month_index] = np.sum( month_of_baseline_seizure_counts, 1 )

    # calculate the monthly seizure frequency for each patient
    baseline_monthly_seizure_frequencies = np.mean(baseline_monthly_seizure_diaries, 1)

    # initialize the array which will hold the times-to-prerandomization
    TTP_times = np.zeros(num_patients)

    # for each patient:
    for patient_index in range(num_patients):
    
        # get the specific diary from the testing period for one patient
        testing_daily_seizure_diary = testing_daily_seizure_diaries[patient_index, :]

        # get the baseline seizure frequency corresponding to that patient
        baseline_monthly_seizure_frequency = baseline_monthly_seizure_frequencies[patient_index]
    
        # initialize a boolean flag indicating whether or not the patient's cumulative seizure count has reached their prerandomization count yet
        reached_count_yet = False
    
        # initialize an index to keep track of which day the following while loop is on
        day_index = 0
    
        # keep track of the cumulaitve count of all the daily seizure counts so far
        sum_count = 0

        # while the patient's cumulative count has no yet reached their prerandomization time:P
        while(not reached_count_yet):
        
            # iterate the cumulative count for the current day
            sum_count = sum_count + testing_daily_seizure_diary[day_index]
        
            # check if  the patient has either reached their prerandomization count or reached the end of their testing period
            reached_count_yet = ( sum_count >= baseline_monthly_seizure_frequency ) or ( day_index == (num_testing_days - 1) )
    
            # iterate the day index
            day_index = day_index + 1
        
        # store the day on which the patient reached their prerandomization time
        TTP_times[patient_index] = day_index
    
    return TTP_times


def generate_one_trial_population(daily_mean, daily_std_dev, rct_params_daily_scale, effect_params):
    '''

    Purpose:

        This function generates the patient diaries needed for one trial which is assumed to have a

        drug arm and a placebo arm. The patient populations of both the drug arm and the placebo arm

        both have the same number of patients.

    Inputs:

        1) daily_mean:

            (float) - the mean of the daily seizure counts in each patient's seizure diary

        2) daily_std_dev:

            (float) - the standard deviation of the daily seizure counts in each patient's seizure diary
        
        3) rct_params_daily_scale:

            (1D Numpy array) - a numpy array containing the RCT design parameters for the trial to be generated

                               on a daily time scale. This Numpy array contains the folllowing quantities:

                                    3.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    3.b) num_baseline_days:

                                        (int) - the number of baseline days in each patient's seizure diary

                                    3.c) num_testing_days:

                                        (int) - the number of testing days in each patient's seizure diary

                                    3.d) min_req_base_sz_count:
        
                                        (int) - the minimum number of required baseline seizure counts

        4) effect_params:

            (1D Numpy array) - a numpy array containing the statistical parameters needed to generate the placebo
                               
                               and drug effects according to the implemented algorithms. Ths numpy array contains

                               the following quantities:

                                    4.a) placebo_mu:

                                        (float) - the mean of the placebo effect

                                    4.b) placebo_sigma:
        
                                        (float) - the standard deviation of the placebo effect

                                    4.c) drug_mu:

                                        (float) - the mean of the drug effect

                                    4.d) drug_sigma:

                                        (float) - the standard deviation of the drug effect

    Outputs:

        1) placebo_arm_daily_seizure_diaries:
        
            (2D Numpy array) - an array of the patient diaries from the placebo arm of one trial

        2) drug_arm_daily_seizure_diaries:

            (2D Numpy array) - an array of the patient diaries from the drug arm of one trial

    '''

    # extract the RCT design parameters
    num_patients_per_trial_arm = rct_params_daily_scale[0]
    num_baseline_days          = rct_params_daily_scale[1]
    num_testing_days           = rct_params_daily_scale[2]
    min_req_base_sz_count      = rct_params_daily_scale[3]

    # extract the statistical parameters for the drug and placebo effect
    placebo_mu    = effect_params[0]
    placebo_sigma = effect_params[1]
    drug_mu       = effect_params[2]
    drug_sigma    = effect_params[3]
    
    # generate all the seizure diaries for the placebo arm
    placebo_arm_daily_seizure_diaries = \
        generate_daily_seizure_diaries(daily_mean, daily_std_dev, num_patients_per_trial_arm, 
                                       num_baseline_days, num_testing_days, 
                                       min_req_base_sz_count)

    # modify the placebo arm diaries via the placebo effect
    placebo_arm_daily_seizure_diaries = \
        apply_effect(placebo_mu, placebo_sigma, placebo_arm_daily_seizure_diaries,
                     num_patients_per_trial_arm, num_baseline_days, num_testing_days)

    # generate all the seizure diaries for the drug arm
    drug_arm_daily_seizure_diaries = \
        generate_daily_seizure_diaries(daily_mean, daily_std_dev, num_patients_per_trial_arm, 
                                       num_baseline_days, num_testing_days, 
                                       min_req_base_sz_count)

    # modify the placebo arm diaries via the placebo effect
    drug_arm_daily_seizure_diaries = \
        apply_effect(placebo_mu, placebo_sigma, drug_arm_daily_seizure_diaries,
                     num_patients_per_trial_arm, num_baseline_days, num_testing_days)

    # modify the placebo arm diaries via the drug effect, after they've already been modified by the placebo effect
    drug_arm_daily_seizure_diaries = \
        apply_effect(drug_mu, drug_sigma, drug_arm_daily_seizure_diaries,
                     num_patients_per_trial_arm, num_baseline_days, num_testing_days)
    
    return [placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries]


def calculate_trial_endpoints(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                              num_patients_per_trial_arm, num_baseline_months, num_testing_months):
    '''

    Purpose:

        This function calculates the endpoint responses (50% responder rate, median percent change, 
    
        time-to-prerandomization) as well as the corresponding p-values for one trial with a placebo arm

        and a drug arm. This function has to be given two sets of seizure diaries, with each set coming from

        the different arms.

    Inputs:

        1) placebo_arm_daily_seizure_diaries:
        
            (2D Numpy Array) - the array of daily seizure diaries from the placebo arm in the form of a 
            
                               2D Numpy array, the first dimension corresponds to the patient number while 
                               
                               the second dimension refers to the days of the seizure count being accessed

        2) drug_arm_daily_seizure_diaries:
        
            (2D Numpy Array) - the array of daily seizure diaries from the drug arm in the form of a 
            
                               2D Numpy array, the first dimension corresponds to the patient number while 
                               
                               the second dimension refers to the days of the seizure count being accessed

        3) num_baseline_months:

            (int) - the number of months in the baseline period of each seizure diary
        
        4) num_testing_months:

            (int) - the number of months in the testing period of each seizure diary

    Outputs:

        1) trial_endpoints:

            (1D Numpy array) - A Numpy array containing the endpoint responses from the placebo and drug arms
                               
                               as well as the corresponding p-values. THis array contains the following

                               quantities:

                                    1.a) placebo_RR50:    
        
                                        (float) - the 50% responder rate for the placebo arm of the trial

                                    1.b) drug_RR50:    
        
                                        (float) - the 50% responder rate for the drug arm of the trial

                                    1.c) RR50_p_value:
        
                                        (float) - the p-value for the 50% responder rate

                                    1.d) placebo_MPC:

                                        (float) - the median percent change for the placebo arm of the trial

                                    1.e) drug_MPC:
        
                                        (float) - the median percent change for the drug arm of the trial

                                    1.f) MPC_p_value:
                
                                        (float) - the p-value for the median percent change

                                    1.g) placebo_med_TTP: 
                
                                        (float) - the median time-to-prerandomization for the placebo arm of the trial

                                    1.h) drug_med_TTP: 
                        
                                        (float) - the median time-to-prerandomization for the drug arm of the trial

                                    1.i) TTP_p_value:
                
                                        (float) - the p-value for the time-to-prerandomization

    '''

    # calculate the number of days in the baseline and testing periods based on the number of months
    num_baseline_days = 28*num_baseline_months
    num_testing_days = 28*num_testing_months

    # calculate the percent changes and times-to-prerandomization for each patient in boht the placebo and drug arm groups
    placebo_percent_changes = calculate_percent_changes(placebo_arm_daily_seizure_diaries, num_baseline_days, num_patients_per_trial_arm)
    drug_percent_changes = calculate_percent_changes(drug_arm_daily_seizure_diaries, num_baseline_days, num_patients_per_trial_arm)
    placebo_TTP_times = calculate_times_to_prerandomization(placebo_arm_daily_seizure_diaries, num_baseline_months, num_testing_days, num_patients_per_trial_arm)
    drug_TTP_times = calculate_times_to_prerandomization(drug_arm_daily_seizure_diaries, num_baseline_months, num_testing_days, num_patients_per_trial_arm)

    # construct the contingency table for the Fisher Exact Test of the RR50 endpoint (comparing placebo arm to drug arm for statistical power)
    placebo_50_percent_responders = np.sum(placebo_percent_changes >= 0.5)
    placebo_50_percent_non_responders = num_patients_per_trial_arm - placebo_50_percent_responders
    drug_50_percent_responders = np.sum(drug_percent_changes >= 0.5)
    drug_50_percent_non_responders = num_patients_per_trial_arm - drug_50_percent_responders
    table = np.array([[placebo_50_percent_responders, placebo_50_percent_non_responders],[drug_50_percent_responders, drug_50_percent_non_responders]])

    # calculate and store the 50% responder rate, median percent change, and median time-to-prerandomization for both the placebo and drug arm groups
    placebo_RR50 = 100*placebo_50_percent_responders/num_patients_per_trial_arm
    drug_RR50 = 100*drug_50_percent_responders/num_patients_per_trial_arm
    placebo_MPC = 100*np.median(placebo_percent_changes)
    drug_MPC = 100*np.median(drug_percent_changes)
    placebo_med_TTP = np.median(placebo_TTP_times)
    drug_med_TTP = np.median(drug_TTP_times)

    # the following two arrays are to meant to say to the logrank test for the TTP endpoint that none of the data is censored (i.e., missing)
    events_observed_placebo = np.ones(len(placebo_TTP_times))
    events_observed_drug = np.ones(len(drug_TTP_times))
    TTP_results = logrank_test(placebo_TTP_times, drug_TTP_times, events_observed_placebo, events_observed_drug)

    # calculate the p-values for the 50% responder rate, the median percent change and the time-to-prerandomization (statistical power)
    [_, RR50_p_value] = stats.fisher_exact(table)
    [_, MPC_p_value] = stats.ranksums(placebo_percent_changes, drug_percent_changes)
    TTP_p_value = TTP_results.p_value

    trial_endpoints = np.array([placebo_RR50,    drug_RR50,    RR50_p_value,
                                placebo_MPC,     drug_MPC,     MPC_p_value,
                                placebo_med_TTP, drug_med_TTP, TTP_p_value])

    return trial_endpoints


def estimate_endpoint_statistics(monthly_mean, monthly_std_dev, num_trials, rct_params_monthly_scale, effect_params):
    '''

    Purpose:

        This function estimates what the expected placebo arm response, the expected drug arm response, 
    
        the statistical power, and the type-1 error should be for a patient over all endpoints (50% responder rate,
    
        median percent change, time-to-prerandomization) with a monthly mean and monthly standard deviation as 
    
        specified by the input parameters. These endpoint statistics are esimated and averaged from a set of simulated trials,
        
        the size of which is specified by the user. This function will just return NaN for all endpoint statistics if the
        
        standard deviation is less than the square root of the mean due to mathematical restrictions on the negative binomial 
        
        distribution which is generating all these seizure counts. This function will also return NaN if the given monthly mean 
        
        is just zero.

    Inputs:

        1) monthly_mean:

            (float) - the mean of the monthly seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the monthly seizure counts in each patient's seizure diary
    
        3) num_trials:

            (int) - the number of trials used to estimate the expected endpoints at each point in the expected 
                    
                    placebo response maps
        
        3) rct_params_monthly_scale:

            (1D Numpy Array) - A Numpy array containing the RCT design parameters (on a monthly time scale) for
                               
                               for all the trials that the endpoint statistics will be estimated over. This array

                               contains the following quantities:

                                    3.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    3.b) num_baseline_months:

                                        (int) - the number of baseline months in each patient's seizure diary

                                    3.c) num_testing_months:

                                        (int) - the number of testing months in each patient's seizure diary

                                    3.d) min_req_base_sz_count:
        
                                        (int) - the minimum number of required baseline seizure counts
        
        4) effect_params:

            (1D Numpy array) - a numpy array containing the statistical parameters needed to generate the placebo
                               
                               and drug effects according to the implemented algorithms. Ths numpy array contains

                               the following quantities:

                                    4.a) placebo_mu:

                                        (float) - the mean of the placebo effect

                                    4.b) placebo_sigma:
        
                                        (float) - the standard deviation of the placebo effect

                                    4.c) drug_mu:

                                        (float) - the mean of the drug effect

                                    4.d) drug_sigma:

                                        (float) - the standard deviation of the drug effect

    Outputs:

        1) endpoint_statistics:

            (1D Numpy array) - A Numpy array which contains the endpoint statistics for one point on all corresponding
                               
                               endpoint statistic maps, estimated over multiple trials. This array contains the following

                               quantities:

                                    1.a) expected_placebo_RR50:

                                        (float) - the 50% responder rate which is expected from an individual in the placebo arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.b) expected_placebo_MPC:

                                        (float) - the median percent change which is expected from an individual in the placebo arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.c) expected_placebo_TTP: 

                                        (float) - the time-to-prerandomization which is expected from an individual in the placebo arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.d) expected_drug_RR50:

                                        (float) - the 50% responder rate which is expected from an individual in the drug arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.e) expected_drug_MPC:

                                        (float) - the median percent change which is expected from an individual in the drug arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.f) expected_drug_TTP:

                                        (float) - the time-to-prerandomization which is expected from an individual in the drug arm with this 
            
                                                  specific monthly mean and monthly standard deviation

                                    1.g) RR50_power:

                                        (float) - the statistical power of the 50% responder rate endpoint for a given monthly seizure frequency 
                      
                                                  and monthly standard deviation

                                    1.h) MPC_power:

                                        (float) - the statistical power of the median percent change endpoint for a given monthly seizure 
            
                                                  frequency and monthly standard deviation

                                    1.i) TTP_power:

                                        (float) - the statistical power of the time-to-prerandomization endpoint for a given monthly seizure 
            
                                                  frequency and monthly standard deviation

                                    1.j) RR50_type_1_error:

                                        (float) - the type-1 error of the 50% responder rate endpoint for a given monthly seizure frequency and 
            
                                                  monthly standard deviation

                                    1.k) MPC_type_1_error:

                                        (float) - the type-1 error of the median percent change endpoint for a given monthly seizure frequency 
                      
                                                  and monthly standard deviation

                                    1.l) TTP_type_1_error:

                                        (float) - the statistical power of the time-to-prerandomization endpoint for a given monthly seizure 
                      
                                                  frequency and monthly standard deviation

    '''

    # extract the RCT design parameters
    num_patients_per_trial_arm = rct_params_monthly_scale[0] 
    num_baseline_months        = rct_params_monthly_scale[1] 
    num_testing_months         = rct_params_monthly_scale[2] 
    min_req_base_sz_count      = rct_params_monthly_scale[3]

    # do a deep copy of the effect_params array that will be used to generate trials with an actual drug effect
    effect_params_placebo_vs_drug = effect_params.copy()

    # create another array of drug and placebo effect parameters, expect the drug effect parameters will be zero,
    # turning the drug arm into another placebo arm
    effect_params_placebo_vs_placebo = np.zeros(4)
    effect_params_placebo_vs_placebo[0] = effect_params[0]
    effect_params_placebo_vs_placebo[1] = effect_params[1]

    # make sure that the patient does not have a true mean seizure count of 0
    if(monthly_mean != 0):

        # make sure that the patient is supposed to have overdispersed data
        if(monthly_std_dev > np.sqrt(monthly_mean)):

            # convert the monthly mean and monthly standard deviation into a daily mean and daily standard deviation
            daily_mean = monthly_mean/28
            daily_std_dev = monthly_std_dev/(28**0.5)
    
            # convert the the number of baseline months and testing months into baseline days and testing days
            num_baseline_days = 28*num_baseline_months
            num_testing_days = 28*num_testing_months

            # initialize the arrays that will contain the endpoint responses (50% responder rate, median percent change, time-to-prerandomization) for both the placebo and drug arms
            placebo_RR50_array = np.zeros(num_trials)
            drug_RR50_array = np.zeros(num_trials)
            placebo_MPC_array = np.zeros(num_trials)
            drug_MPC_array = np.zeros(num_trials)
            placebo_med_TTP_array = np.zeros(num_trials)
            drug_med_TTP_array = np.zeros(num_trials)

            # initialize the arrays that will contain the p_values for the 50% responder rates, median percent changes, and time-to-prerandomization from every trial
            RR50_p_value_array = np.zeros(num_trials)
            MPC_p_value_array = np.zeros(num_trials)
            TTP_p_value_array = np.zeros(num_trials)
            pvp_RR50_p_value_array = np.zeros(num_trials)
            pvp_MPC_p_value_array = np.zeros(num_trials)
            pvp_TTP_p_value_array = np.zeros(num_trials)

            # set the array that will store the RCT design parameters on the daily time scales
            rct_params_daily_scale = np.zeros(4)
            rct_params_daily_scale[0] = num_patients_per_trial_arm
            rct_params_daily_scale[1] = num_baseline_days
            rct_params_daily_scale[2] = num_testing_days
            rct_params_daily_scale[3] = min_req_base_sz_count
            rct_params_daily_scale = np.int_(rct_params_daily_scale)

            # for every trial:
            for trial_index in range(num_trials):

                # generate one set of daily seizure diaries for each placebo arm and drug arm of a trial (two sets in total)
                [placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries] = \
                    generate_one_trial_population(daily_mean, daily_std_dev, rct_params_daily_scale, effect_params_placebo_vs_drug)
                
                # generate two sets of daily seizure diaries, both from a placebo arm (meant for calculating type 1 Error)
                [first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries] = \
                    generate_one_trial_population(daily_mean, daily_std_dev, rct_params_daily_scale, effect_params_placebo_vs_placebo)

                # calculate the 50% responder rate, median percent change, and median time-to-prerandomization for both the placebo and drug arm groups
                trial_endpoints = calculate_trial_endpoints(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                                                            num_patients_per_trial_arm, num_baseline_months, num_testing_months)

                # store the 50% responder rate, median percent change, and median time-to-prerandomization for both the placebo and drug arm groups as well as the corresponding p-values
                placebo_RR50_array[trial_index]    = trial_endpoints[0]
                drug_RR50_array[trial_index]       = trial_endpoints[1]
                RR50_p_value_array[trial_index]    = trial_endpoints[2]
                placebo_MPC_array[trial_index]     = trial_endpoints[3]
                drug_MPC_array[trial_index]        = trial_endpoints[4]
                MPC_p_value_array[trial_index]     = trial_endpoints[5]
                placebo_med_TTP_array[trial_index] = trial_endpoints[6]
                drug_med_TTP_array[trial_index]    = trial_endpoints[7]
                TTP_p_value_array[trial_index]     = trial_endpoints[8]

                # calculate the p-values of the endpoints when two placebo arms are compared against each other
                pvp_trial_endpoints = calculate_trial_endpoints(first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries,
                                                                num_patients_per_trial_arm, num_baseline_months, num_testing_months)

                # store the p-values of the comparison between two placebo arms
                pvp_RR50_p_value_array[trial_index] = pvp_trial_endpoints[2]
                pvp_MPC_p_value_array[trial_index]  = pvp_trial_endpoints[5]
                pvp_TTP_p_value_array[trial_index]  = pvp_trial_endpoints[8]
            
            # calculate the expected endpoint response for both the placebo and drug arms
            expected_placebo_RR50 = np.mean(placebo_RR50_array)
            expected_placebo_MPC = np.mean(placebo_MPC_array)
            expected_placebo_TTP = np.mean(placebo_med_TTP_array)
            expected_drug_RR50 = np.mean(drug_RR50_array)
            expected_drug_MPC = np.mean(drug_MPC_array)
            expected_drug_TTP = np.mean(drug_med_TTP_array)
    
            # calculate the statistical power of each endpoint from all the different p-values calculated
            RR50_power = np.sum(RR50_p_value_array < 0.05)/num_trials
            MPC_power = np.sum(MPC_p_value_array < 0.05)/num_trials
            TTP_power = np.sum(TTP_p_value_array < 0.05)/num_trials

            # calculate the type-1 error of each endpoint from all the different p-values calculated of the two placebo groups
            RR50_type_1_error = np.sum(pvp_RR50_p_value_array < 0.05)/num_trials
            MPC_type_1_error = np.sum(pvp_MPC_p_value_array < 0.05)/num_trials
            TTP_type_1_error = np.sum(pvp_TTP_p_value_array < 0.05)/num_trials

            endpoint_statistics = np.array([expected_placebo_RR50, expected_placebo_MPC, expected_placebo_TTP, 
                                            expected_drug_RR50,    expected_drug_MPC,    expected_drug_TTP,
                                            RR50_power,            MPC_power,            TTP_power, 
                                            RR50_type_1_error,     MPC_type_1_error,     TTP_type_1_error])
        
        # if the patient does not have overdispersed data:
        else:

            # say that calculating their placebo response is impossible
            endpoint_statistics = np.array([np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan])
    
    # if the patient does not have a true mean seizure count of 0, then:
    else:

        # say that calculating their placebo response is impossible
        endpoint_statistics = np.array([np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan])
    
    return endpoint_statistics


def store_endpoint_statistic_map_point(directory, monthly_mean, monthly_std_dev, min_req_base_sz_count, endpoint_statistics):
    '''

    Purpose:

        This function stores all the endpoint statistics for one point on several different type os enpdoint statistic maps.
    
    Inputs:

        1) directory:

            (str) - 

        2) monthly_mean:

            (float) - 
        
       3)  monthly_std_dev:

            (float) - 

        4) min_req_base_sz_count:

            (int) - 
        
        5) trial_endpoints:

            (1D Numpy array) - A Numpy array containing the endpoint responses from the placebo and drug arms
                               
                               as well as the corresponding p-values. THis array contains the following

                               quantities:

                                    5.a) placebo_RR50:    
        
                                        (float) - the 50% responder rate for the placebo arm of the trial

                                    5.b) drug_RR50:    
        
                                        (float) - the 50% responder rate for the drug arm of the trial

                                    5.c) RR50_p_value:
        
                                        (float) - the p-value for the 50% responder rate

                                    5.d) placebo_MPC:

                                        (float) - the median percent change for the placebo arm of the trial

                                    5.e) drug_MPC:
        
                                        (float) - the median percent change for the drug arm of the trial

                                    5.f) MPC_p_value:
                
                                        (float) - the p-value for the median percent change

                                    5.g) placebo_med_TTP: 
                
                                        (float) - the median time-to-prerandomization for the placebo arm of the trial

                                    5.h) drug_med_TTP: 
                        
                                        (float) - the median time-to-prerandomization for the drug arm of the trial

                                    5.i) TTP_p_value:
                
                                        (float) - the p-value for the time-to-prerandomization

    Outputs:

        Tecnically, none, but this function generates a JSON file which is sotred in a file tree structure

    '''

    # turn all relevant information into strings
    monthly_mean_str = str(monthly_mean)
    monthly_std_dev_str = str(monthly_std_dev)
    min_req_base_sz_count_str = str(min_req_base_sz_count)

    # create the file path
    folder = directory + '/eligibility_criteria__' + min_req_base_sz_count_str + '/mean__'  + monthly_mean_str
    file_name = 'std_dev__' + monthly_std_dev_str + '.json'
    file_path = folder + '/' + file_name

    # if the folder which will contain the data does not exist, create it
    if(not os.path.exists(folder) ):

        os.makedirs(folder)
    
    # dump this endpoint statistist map point into a JSON file
    with open(file_path, 'w+') as json_file:

        json.dump(endpoint_statistics.tolist(), json_file)


if(__name__=='__main__'):

    start_time_in_second = time.time()

    # obtain the statistical parameters needed to generate each patient
    monthly_mean    = float(sys.argv[1])
    monthly_std_dev = float(sys.argv[2])

    # obtain the number of trials to estimated the endpoint statistics over
    num_trials = int(sys.argv[3])

    # obtain the directoy in which all the files will be stored
    directory = sys.argv[4]

    # obtain the RCT design parameters
    num_patients_per_trial_arm = int(sys.argv[5])
    num_baseline_months        = int(sys.argv[6])
    num_testing_months         = int(sys.argv[7])
    min_req_base_sz_count      = int(sys.argv[8])

    # obtain the statistical parameters detailing the placebo and drug effects
    placebo_mu    = float(sys.argv[9])
    placebo_sigma = float(sys.argv[10])
    drug_mu       = float(sys.argv[11])
    drug_sigma    = float(sys.argv[12])

    # store some of the parameters into arrays for the sake of readability
    rct_params_monthly_scale = np.array([num_patients_per_trial_arm, num_baseline_months, 
                                         num_testing_months,         min_req_base_sz_count])
    effect_params = np.array([placebo_mu, placebo_sigma, drug_mu, drug_sigma])

    # generate the actual ednpoint statistics
    endpoint_statistics = \
        estimate_endpoint_statistics(monthly_mean, monthly_std_dev, num_trials, rct_params_monthly_scale, effect_params)

    # store the endpoint statistics
    store_endpoint_statistic_map_point(directory, monthly_mean, monthly_std_dev, min_req_base_sz_count, endpoint_statistics)

    #----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------#

    stop_time_in_seconds = time.time()
    run_time_in_minutes = (stop_time_in_seconds - start_time_in_second)/60

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)

    resource_str = '\n\nmonthly mean: '                                  + str(np.round(monthly_mean, 3)) + \
                   '\nmonthly standard deviation: '                      + str(np.round(monthly_std_dev, 3)) + \
                   '\nminimum required number of seizures in baseline: ' + str(np.round(min_req_base_sz_count, 3)) + \
                   '\ncpu time in minutes: '                             + str(np.round(run_time_in_minutes, 3)) + \
                   '\nmemory usage in GB: '                              + str(np.round(used_mem_in_gigabytes, 3))

    file_path = os.getcwd() + '/resource_usage.txt'

    with open(file_path, 'a+') as text_file:

        text_file.write(resource_str)
    
    print(resource_str)

    #----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------#
