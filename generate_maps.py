import numpy as np
from scipy import stats
from lifelines.statistics import logrank_test
import os
import json
import pandas as pd
import time
import argparse
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


def generate_one_trial_population(daily_mean, daily_std_dev, num_patients_per_trial_arm,
                                  num_baseline_days, num_testing_days, min_req_base_sz_count,
                                  placebo_mu, placebo_sigma, drug_mu, drug_sigma):
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

        3) num_patients_per_trial_arm:

            (int) - the number of patients generated per trial arm

        4) num_baseline_days:

            (int) - the number of baseline days in each patient's seizure diary

        5) num_testing_days:

            (int) - the number of testing days in each patient's seizure diary

        6) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts

        7) placebo_mu:

            (float) - the mean of the placebo effect

        8) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        9) drug_mu:

            (float) - the mean of the drug effect

        10) drug_sigma:

            (float) - the standard deviation of the drug effect

    Outputs:

        1) placebo_arm_daily_seizure_diaries:
        
            (2D Numpy array) - an array of the patient diaries from the placebo arm of one trial

        2) drug_arm_daily_seizure_diaries:

            (2D Numpy array) - an array of the patient diaries from the drug arm of one trial

    '''
    
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


def calculate_one_trial_quantities(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                                   num_baseline_months, num_testing_months):
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

        1) placebo_RR50:    
        
            (float) - the 50% responder rate for the placebo arm of the trial

        2) drug_RR50:    
        
            (float) - the 50% responder rate for the drug arm of the trial

        3) RR50_p_value:
        
            (float) - the p-value for the 50% responder rate

        4) placebo_MPC:

            (float) - the median percent change for the placebo arm of the trial

        5) drug_MPC:
        
            (float) - the median percent change for the drug arm of the trial

        6) MPC_p_value:
                
            (float) - the p-value for the median percent change

        7) placebo_med_TTP: 
                
            (float) - the median time-to-prerandomization for the placebo arm of the trial

        8) drug_med_TTP: 
                        
            (float) - the median time-to-prerandomization for the drug arm of the trial

        9) TTP_p_value:
                
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

    return [placebo_RR50,    drug_RR50,    RR50_p_value,
            placebo_MPC,     drug_MPC,     MPC_p_value,
            placebo_med_TTP, drug_med_TTP, TTP_p_value]


def estimate_endpoint_statistics(monthly_mean, monthly_std_dev,
                                 num_patients_per_trial_arm, num_trials,
                                 num_baseline_months, num_testing_months, min_req_base_sz_count,
                                 placebo_mu, placebo_sigma, drug_mu, drug_sigma):
    '''

    Purpose:

        This function estimates what the expected placebo arm response, the expected drug arm response, 
    
        the statistical power, and the type-1 error should be for a patient over all endpoints (50% responder rate,
    
        median percent change, time-to-prerandomization) with a monthly mean and monthly standard deviation as 
    
        specified by the input parameters. This function will just return NaN if the standard deviation is less than 
    
        the square root of the mean due to mathematical restrictions on the negative binomial distribution which is 
    
        generating all these seizure counts. This function will also return NaN if the given monthly mean is just zero.

    Inputs:

        1) monthly_mean:

            (float) - the mean of the monthly seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the monthly seizure counts in each patient's seizure diary

        3) num_patients_per_trial_arm:

            (int) - the number of patients generated per trial arm
        
        4) num_trials:

            (int) - the number of trials used to estimate the expected endpoints at each point in the expected 
                    
                    placebo response maps

        5) num_baseline_months:

            (int) - the number of baseline months in each patient's seizure diary

        6) num_testing_months:

            (int) - the number of testing months in each patient's seizure diary

        7) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts

        8) placebo_mu:

            (float) - the mean of the placebo effect

        9) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        10) drug_mu:

            (float) - the mean of the drug effect

        11) drug_sigma:

            (float) - the standard deviation of the drug effect

    Outputs:

        1) expected_placebo_RR50:

            (float) - the 50% responder rate which is expected from an individual in the placebo arm with this 
            
                      specific monthly mean and monthly standard deviation

        2) expected_placebo_MPC:

            (float) - the median percent change which is expected from an individual in the placebo arm with this 
            
                      specific monthly mean and monthly standard deviation

        3) expected_placebo_TTP: 

            (float) - the time-to-prerandomization which is expected from an individual in the placebo arm with this 
            
                      specific monthly mean and monthly standard deviation

        4) expected_drug_RR50:

            (float) - the 50% responder rate which is expected from an individual in the drug arm with this 
            
                      specific monthly mean and monthly standard deviation

        5) expected_drug_MPC:

            (float) - the median percent change which is expected from an individual in the drug arm with this 
            
                      specific monthly mean and monthly standard deviation

        6) expected_drug_TTP:

            (float) - the time-to-prerandomization which is expected from an individual in the drug arm with this 
            
                      specific monthly mean and monthly standard deviation

        7) RR50_power:

            (float) - the statistical power of the 50% responder rate endpoint for a given monthly seizure frequency 
                      
                      and monthly standard deviation

        8) MPC_power:

            (float) - the statistical power of the median percent change endpoint for a given monthly seizure 
            
                      frequency and monthly standard deviation

        9) TTP_power:

            (float) - the statistical power of the time-to-prerandomization endpoint for a given monthly seizure 
            
                      frequency and monthly standard deviation

        10) RR50_type_1_error:

            (float) - the type-1 error of the 50% responder rate endpoint for a given monthly seizure frequency and 
            
                      monthly standard deviation

        11) MPC_type_1_error:

            (float) - the type-1 error of the median percent change endpoint for a given monthly seizure frequency 
                      
                      and monthly standard deviation

        12) TTP_type_1_error:

            (float) - the statistical power of the time-to-prerandomization endpoint for a given monthly seizure 
                      
                      frequency and monthly standard deviation

    '''


    # make sure that the patient does not have a true mean seizure count of 0
    if(monthly_mean != 0):

        # make sure that the patient is supposed to have overdispersed data
        if(monthly_std_dev > np.sqrt(monthly_mean)):

            # hard-code the number of days in one month into this program
            num_days_in_one_month = 28

            # convert the monthly mean and monthly standard deviation into a daily mean and daily standard deviation
            daily_mean = monthly_mean/num_days_in_one_month
            daily_std_dev = monthly_std_dev/(num_days_in_one_month**0.5)
    
            # convert the the number of baseline months and testing months into baseline days and testing days
            num_baseline_days = num_days_in_one_month*num_baseline_months
            num_testing_days = num_days_in_one_month*num_testing_months

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

            # for every trial:
            for trial_index in range(num_trials):

                # generate one set of daily seizure diaries for each placebo arm and drug arm of a trial (two sets in total)
                [placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries] = \
                    generate_one_trial_population(daily_mean, daily_std_dev, num_patients_per_trial_arm,
                                                  num_baseline_days, num_testing_days, min_req_base_sz_count,
                                                  placebo_mu, placebo_sigma, drug_mu, drug_sigma)
                
                # generate two sets of daily seizure diaries, both from a placebo arm (meant for calculating type 1 Error)
                [first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries] = \
                    generate_one_trial_population(daily_mean, daily_std_dev, num_patients_per_trial_arm,
                                                  num_baseline_days, num_testing_days, min_req_base_sz_count,
                                                  placebo_mu, placebo_sigma, 0, 0)

                # calculate the 50% responder rate, median percent change, and median time-to-prerandomization for both the placebo and drug arm groups
                [placebo_RR50,    drug_RR50,    RR50_p_value,
                 placebo_MPC,     drug_MPC,     MPC_p_value,
                 placebo_med_TTP, drug_med_TTP, TTP_p_value] = \
                     calculate_one_trial_quantities(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                                                   num_baseline_months, num_testing_months)

                # store the 50% responder rate, median percent change, and median time-to-prerandomization for both the placebo and drug arm groups as well as the corresponding p-values
                placebo_RR50_array[trial_index] = placebo_RR50
                drug_RR50_array[trial_index] = drug_RR50
                RR50_p_value_array[trial_index] = RR50_p_value
                placebo_MPC_array[trial_index] = placebo_MPC
                drug_MPC_array[trial_index] = drug_MPC
                MPC_p_value_array[trial_index] = MPC_p_value
                placebo_med_TTP_array[trial_index] = placebo_med_TTP
                drug_med_TTP_array[trial_index] = drug_med_TTP
                TTP_p_value_array[trial_index] = TTP_p_value

                # calculate the p-values of the endpoints when two placebo arms are compared against each other
                [_,    _,    pvp_RR50_p_value,
                 _,    _,    pvp_MPC_p_value,
                 _,    _,    pvp_TTP_p_value] = \
                     calculate_one_trial_quantities(first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries,
                                                   num_baseline_months, num_testing_months)

                # store the p-values of the comparison between two placebo arms
                pvp_RR50_p_value_array[trial_index] = pvp_RR50_p_value
                pvp_MPC_p_value_array[trial_index] = pvp_MPC_p_value
                pvp_TTP_p_value_array[trial_index] = pvp_TTP_p_value
            
            # calculate the expected endpoint response for both the placebo and drug arms
            expected_placebo_RR50 = np.mean(placebo_RR50_array)
            expected_placebo_MPC = np.mean(placebo_MPC_array)
            expected_placebo_TTP = np.mean(placebo_med_TTP_array)
            expected_drug_RR50 = np.mean(drug_RR50_array)
            expected_drug_MPC = np.mean(drug_MPC_array)
            expected_drug_TTP = np.mean(drug_med_TTP_array)
    
            # calculate the statistical power of each endpoint from all the different p-values calculated
            RR50_power = 100*np.sum(RR50_p_value_array < 0.05)/num_trials
            MPC_power  = 100*np.sum(MPC_p_value_array < 0.05)/num_trials
            TTP_power  = 100*np.sum(TTP_p_value_array < 0.05)/num_trials

            # calculate the type-1 error of each endpoint from all the different p-values calculated of the two placebo groups
            RR50_type_1_error = 100*np.sum(pvp_RR50_p_value_array < 0.05)/num_trials
            MPC_type_1_error  = 100*np.sum(pvp_MPC_p_value_array < 0.05)/num_trials
            TTP_type_1_error  = 100*np.sum(pvp_TTP_p_value_array < 0.05)/num_trials

            return [expected_placebo_RR50, expected_placebo_MPC, expected_placebo_TTP, 
                    expected_drug_RR50,    expected_drug_MPC,    expected_drug_TTP,
                    RR50_power,            MPC_power,            TTP_power, 
                    RR50_type_1_error,     MPC_type_1_error,     TTP_type_1_error]
        
        # if the patient does not have overdispersed data:
        else:

            # say that calculating their placebo response is impossible
            return [np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan]
    
    # if the patient does not have a true mean seizure count of 0, then:
    else:

        # say that calculating their placebo response is impossible
        return [np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan]


def generate_endpoint_statistic_maps(start_monthly_mean,         stop_monthly_mean,    step_monthly_mean, 
                                     start_monthly_std_dev,      stop_monthly_std_dev, step_monthly_std_dev,
                                     num_baseline_months,        num_testing_months,   min_req_base_sz_count, 
                                     num_patients_per_trial_arm, num_trials,
                                     placebo_mu, placebo_sigma,  drug_mu, drug_sigma):
    '''

    Purpose:

        This function generates all of the endpoint statistic maps (i.e., expected placebo arm response, expected drug 
    
        response, statistical power, type-1 error) to be stored for later plotting.

    Inputs:

        1) start_monthly_mean:

            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps

        2) stop_monthly_mean:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps

        3) step_monthly_mean:

            (float) - the spaces in between each location on the monthly seizure count mean axis for all expected endpoint maps

        4) start_monthly_std_dev:
        
            (float) - the beginning of the monthly seizure count standard devation axis for all expected endpoint maps

        5) stop_monthly_std_dev:

            (float) - the end of the monthly seizure count tandard devation axis for all expected endpoint maps
        
        6) step_monthly_std_dev:

            (float) - the spaces in between each location on the monthly seizure count standard deviation axis for all expected endpoint maps
        
        7) num_baseline_months:
                
            (int) - the number of baseline months in each patient's seizure diary
        
        8) num_testing_months:
        
            (int) - the number of testing months in each patient's seizure diary
        
        9) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        10) num_patients_per_trial_arm:

            (int) - the number of patients generated per trial arm
        
        11) num_trials:

            (int) -  the number of trials used to estimate the expected endpoints at each point in the map
        
        12) placebo_mu:

            (float) - the mean of the placebo effect

        13) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        14) drug_mu:

            (float) - the mean of the drug effect

        15) drug_sigma:

            (float) - the standard deviation of the drug effect
    
    Outputs:

        1) expected_placebo_RR50_map:

            (2D Numpy array) - a 2D numpy array which contains the expected 50% responder rate placebo response 
                               
                               for many different patients with a given monthly seizure mean and monthly seizure 
                               
                               count standard deviation

        2) expected_placebo_MPC_map:

            (2D Numpy array) - a 2D numpy array which contains the expected median percent change placebo response 
                              
                               for many different patients with a given monthly seizure mean and monthly seizure 
                               
                               count standard deviation

        3) expected_placebo_TTP_map:

            (2D Numpy array) - a 2D numpy array which contains the expected time-to-prerandomization placebo response 
            
                               for many different patients with a given monthly seizure mean and monthly seizure count 
                               
                               standard deviation
        
        4) expected_drug_RR50_map:

            (2D Numpy array) - a 2D numpy array which contains the expected 50% responder rate drug response for many 
            
                               different patients with a given monthly seizure mean and monthly seizure count standard deviation

        5) expected_drug_MPC_map:

            (2D Numpy array) - a 2D numpy array which contains the expected median percent change drug response for many 
            
                               different patients with a given monthly seizure mean and monthly seizure count standard deviation

        6) expected_drug_TTP_map:

            (2D Numpy array) - a 2D numpy array which contains the expected time-to-prerandomization drug response for many 
                               
                               different patients with a given monthly seizure mean and monthly seizure count standard deviation

        7) RR50_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the statistical power of the 50% responder rate for many different 
                               
                               patients with a given monthly seizure mean and monthly seizure count standard deviation

        8) MPC_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the statistical power of the median percent change for many 
                               
                               different patients with a given monthly seizure mean and monthly seizure count standard deviation

        9) TTP_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the statistical power of the time-to-prerandomization for many 
                               
                               different patients with a given monthly seizure mean and monthly seizure count standard deviation
        
        10) RR50_type_1_error_map:

            (2D Numpy array) - a 2D numpy array which contains the type-1 error of the 50% responder rate for many different 
                               
                               patients with a given monthly seizure mean and monthly seizure count standard deviation

        11) MPC_type_1_error_map:

            (2D Numpy array) - a 2D numpy array which contains the type-1 error of the median percent change for many different 
                               
                               patients with a given monthly seizure mean and monthly seizure count standard deviation

        12) TTP_type_1_error_map:

            (2D Numpy array) - a 2D numpy array which contains the type-1 error of the time-to-prerandomization for many different 
                               
                               patients with a given monthly seizure mean and monthly seizure count standard deviation

    '''

    # create the monthly mean and monthly standard deviation axes
    monthly_mean_array = np.arange(start_monthly_mean, stop_monthly_mean + step_monthly_mean, step_monthly_mean)
    monthly_std_dev_array = np.arange(start_monthly_std_dev, stop_monthly_std_dev + step_monthly_std_dev, step_monthly_std_dev)

    # flip the monthly seizure count standard deviation axes
    monthly_std_dev_array = np.flip(monthly_std_dev_array, 0)

    # count up the number of locations on both axes
    num_monthly_means = len(monthly_mean_array)
    num_monthly_std_devs = len(monthly_std_dev_array)

    # initialize the 2D numpy arrays that will hold the endpoint statistics maps
    expected_placebo_RR50_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_placebo_MPC_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_placebo_TTP_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_drug_RR50_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_drug_MPC_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    expected_drug_TTP_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    RR50_stat_power_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    MPC_stat_power_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    TTP_stat_power_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    RR50_type_1_error_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    MPC_type_1_error_map = np.zeros((num_monthly_std_devs, num_monthly_means))
    TTP_type_1_error_map = np.zeros((num_monthly_std_devs, num_monthly_means))

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
            [expected_placebo_RR50, expected_placebo_MPC, expected_placebo_TTP, 
             expected_drug_RR50,    expected_drug_MPC,    expected_drug_TTP,
             RR50_power,            MPC_power,            TTP_power, 
             RR50_type_1_error,     MPC_type_1_error,     TTP_type_1_error] =  \
                estimate_endpoint_statistics(monthly_mean, monthly_std_dev,
                                             num_patients_per_trial_arm, num_trials,
                                             num_baseline_months, num_testing_months, min_req_base_sz_count,
                                             placebo_mu, placebo_sigma, drug_mu, drug_sigma)
            
            # put a stop to the timer on the endpoint estimation
            stop_time_in_seconds = time.time()
            
            # calculate the total number of minutes it took to estimate the expected endpoints for the given mean and standard deviation
            total_time_in_minutes = (stop_time_in_seconds - start_time_in_seconds)/60

            # store the estimated endpoint statistics
            expected_placebo_RR50_map[monthly_std_dev_index, monthly_mean_index] = expected_placebo_RR50
            expected_placebo_MPC_map[monthly_std_dev_index, monthly_mean_index]  = expected_placebo_MPC
            expected_placebo_TTP_map[monthly_std_dev_index, monthly_mean_index]  = expected_placebo_TTP
            expected_drug_RR50_map[monthly_std_dev_index, monthly_mean_index]    = expected_drug_RR50
            expected_drug_MPC_map[monthly_std_dev_index, monthly_mean_index]     = expected_drug_MPC
            expected_drug_TTP_map[monthly_std_dev_index, monthly_mean_index]     = expected_drug_TTP
            RR50_stat_power_map[monthly_std_dev_index, monthly_mean_index]       = RR50_power
            MPC_stat_power_map[monthly_std_dev_index, monthly_mean_index]        = MPC_power
            TTP_stat_power_map[monthly_std_dev_index, monthly_mean_index]        = TTP_power
            RR50_type_1_error_map[monthly_std_dev_index, monthly_mean_index]     = RR50_type_1_error
            MPC_type_1_error_map[monthly_std_dev_index, monthly_mean_index]      = MPC_type_1_error
            TTP_type_1_error_map[monthly_std_dev_index, monthly_mean_index]      = TTP_type_1_error

            # prepare a string telling the user where the algorithm is in terms of map generation
            cpu_time_string = 'cpu time (minutes): ' + str( np.round(total_time_in_minutes, 2) )

            expected_placebo_RR50_string = 'expected placebo RR50: ' + str( np.round(float(expected_placebo_RR50), 4) ) + ' %'
            expected_placebo_MPC_string = 'expected placebo MPC: '   + str( np.round(float(expected_placebo_MPC), 4) )  + ' %'
            expected_placebo_TTP_string = 'expected placebo TTP: '   + str( np.round(float(expected_placebo_TTP), 4) )
            expected_drug_RR50_string = 'expected drug RR50: '       + str( np.round(float(expected_drug_RR50), 4) )    + ' %'
            expected_drug_MPC_string = 'expected drug MPC: '         + str( np.round(float(expected_drug_MPC), 4) )     + ' %'
            expected_drug_TTP_string = 'expected drug TTP: '         + str( np.round(float(expected_drug_TTP), 4) )
            RR50_stat_power_string = 'RR50 stat power:  '            + str( np.round(float(RR50_power), 4) )            + ' %'
            MPC_stat_power_string = 'MPC stat power:  '              + str( np.round(float(MPC_power), 4) )             + ' %'
            TTP_stat_power_string = 'TTP stat power:  '              + str( np.round(float(TTP_power), 4) )             + ' %'
            RR50_type_1_error_string = 'RR50 type-1 error:  '        + str( np.round(float(RR50_type_1_error), 4) )     + ' %'
            MPC_type_1_error_string = 'MPC type-1 error:  '          + str( np.round(float(MPC_type_1_error), 4) )      + ' %'
            TTP_type_1_error_string = 'TTP type-1 error:  '          + str( np.round(float(TTP_type_1_error), 4) )      + ' %'

            monthly_mean_string = str(np.round(monthly_mean, 2))
            monthly_std_dev_string = str(np.round(monthly_std_dev, 2))

            expected_placebo_string = expected_placebo_RR50_string + '\n' + expected_placebo_MPC_string + '\n' + expected_placebo_TTP_string
            expected_drug_string = expected_drug_RR50_string + '\n' + expected_drug_MPC_string + '\n' + expected_drug_TTP_string
            stat_power_string = RR50_stat_power_string + '\n' + MPC_stat_power_string + '\n' + TTP_stat_power_string
            type_1_error_string = RR50_type_1_error_string + '\n' + MPC_type_1_error_string + '\n' + TTP_type_1_error_string

            orientation_string = 'monthly mean, monthly standard deviation: (' + monthly_mean_string + ', ' + monthly_std_dev_string + ')'

            data_string = '\n\n' + orientation_string + ':\n' + expected_placebo_string + '\n' + expected_drug_string + '\n' + stat_power_string + '\n' + type_1_error_string + '\n' + cpu_time_string

            # print the string
            print(data_string)
    
    return [expected_placebo_RR50_map, expected_placebo_MPC_map, expected_placebo_TTP_map,
            expected_drug_RR50_map,    expected_drug_MPC_map,    expected_drug_TTP_map,
            RR50_stat_power_map,       MPC_stat_power_map,       TTP_stat_power_map,
            RR50_type_1_error_map,     MPC_type_1_error_map,     TTP_type_1_error_map]


def store_map(data_map, data_map_file_name,
              directory, min_req_base_sz_count, folder):
    '''

    Purpose:

        This function takes a data map as the parameters used to create its axes (referred to as the metadata)

        and puts all the information into intermediate JSON files which are located in a folder specified by 
    
        the end user.

    Inputs:

        1) data_map:

            (2D Numpy array) - a 2D numpy array containing data to be stored now in a JSON
                               
                               file and plotted later
        
        2) data_map_file_name:

            (string) - the name of the JSON file which will contain the data map to be plotted

                       later
        
        ***********************************************************
        *************BEGINNING OF DEPRECATED INPUTS****************
        ***********************************************************

        3) data_map_meta_data_file_name:

            (string) - the name of the JSON file which will contain the metadata for the data 
            
                       map to be plotted later

        4) x_axis_start:

            (float) - the beginning of the x-axis for the data map to be plotted later
        
        5) x_axis_stop:

            (float) - the end of the x-axis for the data map to be plotted later
        
        6) x_axis_step:

            (float) - the spaces in between each location on the x-axis for all expected endpoint maps
        
        7) y_axis_start:

            (float) - the beginning of the y-axis for the data map to be plotted later
        
        8) y_axis_stop:

            (float) - the end of the y-axis for the data map to be plotted later
        
        9) y_axis_step:

            (float) - the spaces in between each location on the y-axis for all expected endpoint maps

        ***********************************************************
        *****************END OF DEPRECATED INPUTS******************
        ***********************************************************
        
        10) directory:

            (string) - the name of the directory which contains the folder in which all the intermediate JSON files will be stored
        
        11) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        12) folder:

            (string) - the name of the actual folder in which all the intermediate JSON files will be stored
    
    Outputs:

        Technically None

    '''

    # get the file path for the folder specified
    folder_path = directory + '/' + str(min_req_base_sz_count) +  '/' + folder

    # get the file path for the JSON file that will store the data map 
    data_map_file_path = folder_path + '/' + data_map_file_name + '.json'
    
    '''
    # get the file path for the JSON file that will store the data map metadata
    data_map_metadata_file_path = folder_path + '/' + data_map_meta_data_file_name + '.json'

    # put all of the metadata into one array
    metadata = np.array([x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step])
    '''

    # make sure that the specified folder exists if it doesn't already
    if ( not os.path.exists(folder_path) ):

        os.makedirs(folder_path)

    # open/create the JSON file that will store the data map 
    with open(data_map_file_path, 'w+') as map_storage_file:

        # store the data map
        json.dump(data_map.tolist(), map_storage_file)

    '''
    # open/create the JSON file that will store the data map metadata
    with open(data_map_metadata_file_path, 'w+') as map_metadata_storage_file:

        # store the data map
        json.dump(metadata.tolist(), map_metadata_storage_file)
    '''


def main(start_monthly_mean, stop_monthly_mean, step_monthly_mean, 
         start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
         num_baseline_months, num_testing_months, min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
         placebo_mu, placebo_sigma,  drug_mu, drug_sigma, directory, folder):
    '''

    Purpose:

        This function is the main function that should be called in order to fulfill this script's primary purpose,

        which is generating and storing the data needed to plot the endpoint statistic maps to be plotted later. This data

        is stored as intermediate JSON files in a folder specified by the end user.

    Inputs:

        1) num_patients_per_model:
                
            (int) - the number of patients to generate for the histograms of models 1 and 2
        
        2) num_months_per_patient:

            (int) - the number of months that each patient should have in both NV models

        3) start_monthly_mean:
        
            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps=
        
        4) stop_monthly_mean:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps
        
        5) step_monthly_mean:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps

        6) start_monthly_std_dev:
                
            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps
        
        7) stop_monthly_std_dev:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps
        
        8) step_monthly_std_dev:

            (float) - the spaces in between each location on the monthly seizure count mean axis for all expected endpoint maps

        9) num_baseline_months:
                        
            (int) - the number of baseline months in each patient's seizure diary
        
        10) num_testing_months:
                
            (int) - the number of testing months in each patient's seizure diary
        
        11) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        12) num_patients_per_trial_arm:

            (int) - the number of patients generated per trial arm
        
        13) num_trials:

            (int) -  the number of trials used to estimate the expected endpoints at each point in the expected placebo response maps
        
        14) placebo_mu:

            (float) - the mean of the placebo effect

        15) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        16) drug_mu:

            (float) - the mean of the drug effect

        17) drug_sigma:

            (float) - the standard deviation of the drug effect
        
        18) directory:

            (string) - the name of the directory which contains the folder in which all the intermediate JSON files will be stored
        
        19) folder:

            (string) - the name of the actual folder in which all the intermediate JSON files will be stored
    
    Outputs:

        Technically None

    '''

    # set the names of the files that the data/metadata of the endpoint statistic maps will be stored in
    expected_placebo_RR50_file_name           = 'expected_placebo_RR50_map'
    expected_placebo_MPC_file_name            = 'expected_placebo_MPC_map'
    expected_placebo_TTP_file_name            = 'expected_placebo_TTP_map'
    expected_drug_RR50_file_name              = 'expected_drug_RR50_map'
    expected_drug_MPC_file_name               = 'expected_drug_MPC_map'
    expected_drug_TTP_file_name               = 'expected_drug_TTP_map'
    RR50_stat_power_file_name                 = 'RR50_stat_power_map'
    MPC_stat_power_file_name                  = 'MPC_stat_power_map'
    TTP_stat_power_file_name                  = 'TTP_stat_power_map'
    RR50_type_1_error_file_name               = 'RR50_type_1_error_map'
    MPC_type_1_error_file_name                = 'MPC_type_1_error_map'
    TTP_type_1_error_file_name                = 'TTP_type_1_error_map'

    [expected_placebo_RR50_map,     expected_placebo_MPC_map,     expected_placebo_TTP_map,
     expected_drug_RR50_map,        expected_drug_MPC_map,        expected_drug_TTP_map,
     RR50_stat_power_map,           MPC_stat_power_map,           TTP_stat_power_map,
     RR50_type_1_error_map,         MPC_type_1_error_map,         TTP_type_1_error_map      ] = \
        generate_endpoint_statistic_maps(start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
                                         start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
                                         num_patients_per_trial_arm, num_trials,
                                         num_baseline_months, num_testing_months, min_req_base_sz_count,
                                         placebo_mu, placebo_sigma,  drug_mu, drug_sigma)

    # store the expected 50% responder rate placebo arm response map
    store_map(expected_placebo_RR50_map, expected_placebo_RR50_file_name,
              directory, min_req_base_sz_count, folder)
    
    # store the expected median percent change placebo arm response map
    store_map(expected_placebo_MPC_map, expected_placebo_MPC_file_name,
              directory, min_req_base_sz_count, folder)

    # store the expected time-to-prerandomization placebo arm response map
    store_map(expected_placebo_TTP_map, expected_placebo_TTP_file_name,
              directory, min_req_base_sz_count, folder)

    # store the expected 50% responder rate drug arm response map
    store_map(expected_drug_RR50_map, expected_drug_RR50_file_name,
              directory, min_req_base_sz_count, folder)
    
    # store the expected median percent change drug arm response map
    store_map(expected_drug_MPC_map, expected_drug_MPC_file_name,
              directory, min_req_base_sz_count, folder)

    # store the expected time-to-prerandomization drug arm response map
    store_map(expected_drug_TTP_map, expected_drug_TTP_file_name,
              directory, min_req_base_sz_count, folder)

    # store the 50% responder rate statistical power map
    store_map(RR50_stat_power_map, RR50_stat_power_file_name,
              directory, min_req_base_sz_count, folder)

    # store the median percent change statistical power map
    store_map(MPC_stat_power_map, MPC_stat_power_file_name,
              directory, min_req_base_sz_count, folder)
    
    # store the time-to-prerandomization statistical power map
    store_map(TTP_stat_power_map, TTP_stat_power_file_name,
              directory, min_req_base_sz_count, folder)

    # store the 50% responder rate type-1 error map
    store_map(RR50_type_1_error_map, RR50_type_1_error_file_name,
              directory, min_req_base_sz_count, folder)

    # store the median percent change type-1 error map
    store_map(MPC_type_1_error_map, MPC_type_1_error_file_name,
              directory, min_req_base_sz_count, folder)
    
    # store the time-to-prerandomization type-1 error map
    store_map(TTP_type_1_error_map, TTP_type_1_error_file_name,
              directory, min_req_base_sz_count, folder)


if(__name__=='__main__'):
    '''

        The purpose of this if-statement is to act as the main interface between this python

        script and its wrapper shell script (which passes on all the variables specified by
        
        the user). The main() function handles the rest of the algorithm implemented by this script.

    '''
    start_time_in_seconds = time.time()

    # take in the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('array', nargs='+')
    args = parser.parse_args()
    arg_array = args.array

    # obtain the information needed for the x-axis of the endpoint statistic maps
    start_monthly_mean = float(arg_array[0])
    stop_monthly_mean = float(arg_array[1])
    step_monthly_mean = float(arg_array[2])

    # obtain the information needed for the y-axis of the endpoint statistic maps
    start_monthly_std_dev = float(arg_array[3])
    stop_monthly_std_dev = float(arg_array[4])
    step_monthly_std_dev = float(arg_array[5])

    # obtain the parameters for estimating the placebo response at each point on the endpoint statistic maps
    num_baseline_months = int(arg_array[6])
    num_testing_months = int(arg_array[7])
    min_req_base_sz_count = int(arg_array[8])
    num_patients_per_trial_arm = int(arg_array[9])
    num_trials = int(arg_array[10])

    # obtain the parameters for generating the placebo and drug effects
    placebo_mu = float(arg_array[11])
    placebo_sigma = float(arg_array[12])
    drug_mu = float(arg_array[13])
    drug_sigma = float(arg_array[14])

    # obtain the location of the directory containing the folder in which all the intermediate JSON files for this specific map will be stored
    directory = arg_array[15]

    # obtain the name of the actual folder in which all the intermediate JSON files for this specific map will be stored
    folder = arg_array[16]

    # call the main() function
    main(start_monthly_mean, stop_monthly_mean, step_monthly_mean, 
         start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
         num_baseline_months, num_testing_months, min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
         placebo_mu, placebo_sigma,  drug_mu, drug_sigma, directory, folder)
    
    stop_time_in_seconds = time.time()

    total_time_in_minutes  = (stop_time_in_seconds - start_time_in_seconds)/60

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)

    print('\n\ncpu time in minutes: ' + str(total_time_in_minutes) + '\nmemory usage in GB: ' + str(used_mem_in_gigabytes) + '\n' )


