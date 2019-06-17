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

    This function calculates the time-to-prerandomization endpoint for each patient' daily seizure diary.

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
    placebo_RR50 = placebo_50_percent_responders/num_patients_per_trial_arm
    drug_RR50 = drug_50_percent_responders/num_patients_per_trial_arm
    placebo_MPC = np.median(placebo_percent_changes)
    drug_MPC = np.median(drug_percent_changes)
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
                RR50_p_value_array[trial_index] = pvp_RR50_p_value
                MPC_p_value_array[trial_index] = pvp_MPC_p_value
                TTP_p_value_array[trial_index] = pvp_TTP_p_value
            
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

    This function generates all of the 2D endpoint statistic arrays (i.e., expected placebo arm response, expected drug 
    
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

        1) RR50_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the RR50 statistical power map for many different patients with 
            
                               a given monthly seizure mean and monthly seizure count standard deviation

        2) MPC_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the MPC statistical power map for many different patients with 
            
                               a given monthly seizure mean and monthly seizure count standard deviation

        3) TTP_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the TTP statistical power map for many different patients with 
            
                               a given monthly seizure mean and monthly seizure count standard deviation

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
            expected_placebo_RR50_string = 'expected placebo RR50: ' +  str( 100*np.round(expected_placebo_RR50, 4) ) + ' %'
            expected_placebo_MPC_string = 'expected placebo MPC: ' +  str( 100*np.round(expected_placebo_MPC, 4) ) + ' %'
            expected_placebo_TTP_string = 'expected placebo TTP: ' +  str( np.round(expected_placebo_TTP, 4) )
            expected_drug_RR50_string = 'expected drug RR50: ' +  str( 100*np.round(expected_placebo_RR50, 4) ) + ' %'
            expected_drug_MPC_string = 'expected drug MPC: ' +  str( 100*np.round(expected_placebo_MPC, 4) ) + ' %'
            expected_drug_TTP_string = 'expected drug TTP: ' +  str( np.round(expected_drug_TTP, 4) )
            RR50_stat_power_string = 'RR50 stat power:  ' +  str( 100*np.round(RR50_power, 4) ) + ' %'
            MPC_stat_power_string = 'MPC stat power:  ' +  str( 100*np.round(MPC_power, 4) ) + ' %'
            TTP_stat_power_string = 'TTP stat power:  ' + str( 100*np.round(TTP_power, 4) ) + ' %'
            RR50_type_1_error_string = 'RR50 type-1 error:  ' +  str( 100*np.round(RR50_power, 4) ) + ' %'
            MPC_type_1_error_string = 'MPC type-1 error:  ' +  str( 100*np.round(MPC_power, 4) ) + ' %'
            TTP_type_1_error_string = 'TTP type-1 error:  ' + str( 100*np.round(TTP_power, 4) ) + ' %'
            expected_placebo_string = expected_placebo_RR50_string + '\n' + expected_placebo_MPC_string + '\n' + expected_placebo_TTP_string
            expected_drug_string = expected_drug_RR50_string + '\n' + expected_drug_MPC_string + '\n' + expected_drug_TTP_string
            stat_power_string = RR50_stat_power_string + '\n' + MPC_stat_power_string + '\n' + TTP_stat_power_string
            type_1_error_string = RR50_type_1_error_string + '\n' + MPC_type_1_error_string + '\n' + TTP_type_1_error_string
            monthly_mean_string = str(np.round(monthly_mean, 2))
            monthly_std_dev_string = str(np.round(monthly_std_dev, 2))
            orientation_string = 'monthly mean, monthly standard deviation: (' + monthly_mean_string + ', ' + monthly_std_dev_string + ')'
            data_string = '\n\n' + orientation_string + ':\n' + expected_placebo_string + '\n' + expected_drug_string + '\n' + stat_power_string + '\n' + type_1_error_string + '\n' + cpu_time_string

            # print the string
            print(data_string)
    
    return [expected_placebo_RR50_map, expected_placebo_MPC_map, expected_placebo_TTP_map,
            expected_drug_RR50_map,    expected_drug_MPC_map,    expected_drug_TTP_map,
            RR50_stat_power_map,       MPC_stat_power_map,       TTP_stat_power_map,
            RR50_type_1_error_map,     MPC_type_1_error_map,     TTP_type_1_error_map]


def generate_model_patient_data(shape, scale, alpha, beta, num_patients_per_model, num_months_per_patient):
    '''

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


def generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, 
                      shape_2, scale_2, alpha_2, beta_2, 
                      num_patients_per_model, num_months_per_patient,
                      start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
                      start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
                      num_patients_per_trial_arm, num_trials,
                      num_baseline_months, num_testing_months, min_req_base_sz_count,
                      placebo_mu, placebo_sigma,  drug_mu, drug_sigma):

    '''

    This function generates several things: first of all, it generates the statistical power maps for the 50% 
    
    responder rate and the median percent change. Second of all, it generates histograms of all the patients generated by 
    
    NV model 1 and NV model 2. Third of all, it generates the collective placebo response of all the patients generated from 
    
    NV model 1 and NV model 2, which is calculated by summing over the multiplication of the expected endpoint placebo response 
    
    maps and the model 1 and model 2 histograms.

    Inputs:

        1) shape_1:
        
            (float) - the first parameter for NV model 1
        
        2) scale_1:
        
            (float) - the second parameter for NV model 1
        
        3) alpha_1:
        
            (float) - the third parameter for NV model 1
        
        4) beta_1:
        
            (float) - the fourth parameter for NV model 1

        5) shape_2:
        
            (float) - the first parameter for NV model 2
        
        6) scale_2:
        
            (float) - the second parameter for NV model 2

        7) alpha_2:
        
            (float) - the third parameter for NV model 2
        
        8) beta_2:

            (float) -  the fourth parameter for NV model 2
        
        9) num_patients_per_model:
        
            (int) - the number of patients to generate for the histograms of models 1 and 2
        
        10) num_months_per_patient:
        
            (int) - the number of months that each patient should have in both NV models

        12) start_monthly_mean:
        
            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps
            
        13) stop_monthly_mean:
        
            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps
            
        14) step_monthly_mean:
            
            (float) - the spaces in between each location on the monthly seizure count mean axis for all expected endpoint maps

        15) start_monthly_std_dev:
        
            (float) - the beginning of the monthly seizure count standard devation axis for all expected endpoint maps
        
        16) stop_monthly_std_dev:
        
            (float) - the end of the monthly seizure count tandard devation axis for all expected endpoint maps
        
        17) step_monthly_std_dev:
            
            (float) - the spaces in between each location on the monthly seizure count standard deviation axis for all expected endpoint maps

        18) num_patients_per_trial_arm:
        
            (int) -  the number of patients generated per trial arm
            
        19) num_trials:
            
            (int) - the number of trials used to estimate the expected endpoints at each point in the expected endpoint response maps
        
        20) num_baseline_months:
        
            (int) - the number of baseline months in each patient's seizure diary (different than num_months_per_patients, which affects
                    
                    the NV model patients as opposed to the patients used to estimate the expected endpoints)
        
        21) num_testing_months:
            
            (int) - the number of testing months in each patient's seizure diary (different than num_months_per_patients, which affects
                    
                    the NV model patients as opposed to the patients used to estimate the expected endpoints)
        
        22) min_req_base_sz_count:
            
            (int) - the minimum number of required baseline seizure counts for all patients used to estimate the expected placebo response maps
        
        23) placebo_mu:

            (float) - the mean of the placebo effect

        24) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        25) drug_mu:

            (float) - the mean of the drug effect

        26) drug_sigma:

            (float) - the standard deviation of the drug effect

    Outputs:

        1) Model_1_expected_placebo_RR50:
                    
            (float) - the collective expected 50% responder rate placebo arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected 50% responder rate 
                      
                      placebo arm response map
        
        2) Model_1_expected_placebo_MPC:
                    
            (float) - the collective expected median percent change placebo arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected median percent change 
                      
                      placebo arm response map
        
        3) Model_1_expected_placebo_TTP:
                    
            (float) - the collective expected time-to-prerandomization placebo arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected time-to-prerandomization 
                      
                      placebo arm response map
        
        4) Model_2_expected_placebo_RR50:
                    
            (float) - the collective expected 50% responder rate placebo arm response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected 50% responder rate 
                      
                      placebo arm response map
        
        5) Model_2_expected_placebo_MPC:
                            
            (float) - the collective expected median percent change placebo arm response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected median percent change 
                      
                      placebo arm response maps
        
        6) Model_2_expected_placebo_TTP:
                           
            (float) - the collective expected time-to-prerandomization placebo response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected time-to-prerandomization 
                      
                      placebo response map
        
        7) Model_1_expected_drug_RR50:
                            
            (float) - the collective expected 50% responder rate drug arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected 50% responder rate 
                      
                      drug arm response map
        
        8) Model_1_expected_drug_MPC:
                                    
            (float) - the collective expected median percent change drug arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected median percent change 
                      
                      drug arm response map
        
        9) Model_1_expected_drug_TTP:
                                    
            (float) - the collective expected time-to-prerandomization drug arm response for all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the expected time-to-prerandomization 
                      
                      drug arm response map
        
        10) Model_2_expected_drug_RR50:
                                    
            (float) - the collective expected 50% responder rate drug arm response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected 50% responder rate 
                      
                      drug arm response map
        
        11) Model_2_expected_drug_MPC:
                                    
            (float) - the collective expected median percent change drug arm response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected median percent change 
                      
                      drug arm response map
        
        12) Model_2_expected_drug_TTP:
                                            
            (float) - the collective expected time-to-prerandomization drug arm response for all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the expected time-to-prerandomization 
                      
                      drug arm response map
        
        13) Model_1_RR50_stat_power:
                    
            (float) - the collective statistical power for the 50% responder rate over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the 50% responder rate statistical
                      
                      power map
        
        14) Model_1_MPC_stat_power:
                    
            (float) - the collective statistical power for the median percent change over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the median percent change statistical
                      
                      power map
        
        15) Model_1_TTP_stat_power:
                    
            (float) - the collective statistical power for the time-to-prerandomization over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the time-to-prerandomization statistical
                      
                      power map
        
        16)  Model_2_RR50_stat_power:
                    
            (float) - the collective statistical power for the 50% responder rate over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the 50% responder rate statistical
                      
                      power map
        
        17) Model_2_MPC_stat_power:
                    
            (float) - the collective statistical power for the median percent change over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the median percent change statistical
                      
                      power map
        
        18) Model_2_TTP_stat_power:
                    
            (float) - the collective statistical power for the time-to-prerandomization over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the time-to-prerandomization statistical
                      
                      power map
        
        19) Model_1_RR50_type_1_error:
                            
            (float) - the collective type-1 error for the 50% responder rate over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the 50% responder rate type-1 
                      
                      error map
        
        20) Model_1_MPC_type_1_error:
                                    
            (float) - the collective type-1 error for the median percent change over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the median percent change type-1 
                      
                      error map
        
        21) Model_1_TTP_type_1_error:
                            
            (float) - the collective type-1 error for the time-to-prerandomization over all of the patients from model 1, 
            
                      determined by a multiplication of the Model 1 patient histogram and the time-to-prerandomization type-1 
                      
                      error map
        
        22) Model_2_RR50_type_1_error:
                            
            (float) - the collective type-1 error for the 50% responder rate over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the 50% responder rate type-1 
                      
                      error map
        
        23) Model_2_MPC_type_1_error:
                                            
            (float) - the collective type-1 error for the median percent change over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the median percent change type-1 
                      
                      error map
        
        24) Model_2_TTP_type_1_error:
                                    
            (float) - the collective type-1 error for the time-to-prerandomization over all of the patients from model 2, 
            
                      determined by a multiplication of the Model 2 patient histogram and the time-to-prerandomization type-1 
                      
                      error map
        
        25) expected_placebo_RR50_map:
        
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint 50% responder rate placebo response map
        
        26) expected_placebo_MPC_map:
                
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint median percent change placebo response map
        
        27) expected_placebo_TTP_map:
                
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint time-to-prerandomization placebo response map
        
        28) expected_drug_RR50_map:
                
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint 50% responder rate drug response map
        
        29) expected_drug_MPC_map:
                        
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint median percent change drug response map
        
        30) expected_drug_TTP_map:
                                
            (2D Numpy array) - a 2D numpy array which contains the expected endpoint time-to-prerandomization drug response map
        
        31) RR50_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the 50% responder rate statistical power map
        
        32) MPC_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the median percent change statistical power map
        
        33) TTP_stat_power_map:

            (2D Numpy array) - a 2D numpy array which contains the time-to-prerandomization statistical power map
        
        34) RR50_type_1_error_map:
        
            (2D Numpy array) - a 2D numpy array which contains the 50% responder rate type-1 error map
        
        35) MPC_type_1_error_map:
                
            (2D Numpy array) - a 2D numpy array which contains the median percent change type-1 error map
        
        36) TTP_type_1_error_map:
                
            (2D Numpy array) - a 2D numpy array which contains the time-to-prerandomization type-1 error map
        
        37) H_model_1:
            
            (2D Numpy array) - histogram of model 1 patients with the same dimensions and bins as the expected placebo response maps
        
        38) H_model_2:

            (2D Numpy array) - histogram of model 2 patients with the same dimensions and bins as the expected placebo response maps

    '''

    # generate the endpoint statistic maps for RR50, MPC, and TTP
    [expected_placebo_RR50_map, expected_placebo_MPC_map, expected_placebo_TTP_map,
     expected_drug_RR50_map,    expected_drug_MPC_map,    expected_drug_TTP_map,
     RR50_stat_power_map,       MPC_stat_power_map,       TTP_stat_power_map,
     RR50_type_1_error_map,     MPC_type_1_error_map,     TTP_type_1_error_map      ] = \
        generate_endpoint_statistic_maps(start_monthly_mean,         stop_monthly_mean,    step_monthly_mean, 
                                         start_monthly_std_dev,      stop_monthly_std_dev, step_monthly_std_dev,
                                         num_baseline_months,        num_testing_months,   min_req_base_sz_count, 
                                         num_patients_per_trial_arm, num_trials,
                                         placebo_mu, placebo_sigma,  drug_mu, drug_sigma)

    # generate Model 1 patients
    [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = \
                                        generate_model_patient_data(shape_1, scale_1, alpha_1, beta_1,
                                                                    num_patients_per_model, num_months_per_patient)

    # generate Model 2 patients
    [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = \
                                    generate_model_patient_data(shape_2, scale_2, alpha_2, beta_2,
                                                                num_patients_per_model, num_months_per_patient)

    # get the size of those ednpoint statistical power maps
    [nx, ny] = RR50_stat_power_map.shape

    # calculate the histogram of the Model 1 patients
    [H_model_1, _, _] = np.histogram2d(model_1_monthly_count_averages, model_1_monthly_count_standard_deviations, bins=[ny, nx], 
                                        range=[[start_monthly_mean, stop_monthly_mean], [start_monthly_std_dev, stop_monthly_std_dev]])
    H_model_1 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_1))))
    norm_const_1 = np.sum(np.sum(H_model_1, 0))
    H_model_1 = H_model_1/norm_const_1

    # calculate the histogram of the Model 2 patients
    [H_model_2, _, _] = np.histogram2d(model_2_monthly_count_averages, model_2_monthly_count_standard_deviations, bins=[ny, nx], 
                                    range=[[start_monthly_mean, stop_monthly_mean], [start_monthly_std_dev, stop_monthly_std_dev]])
    H_model_2 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_2))))
    norm_const_2 = np.sum(np.sum(H_model_2, 0))
    H_model_2 = H_model_2/norm_const_2

    # combine the expected endpoint maps as well as the Model 1 and Model 2 histograms to get the endpoint statistics of RR50, MPC, and TTP for both models
    Model_1_expected_placebo_RR50 = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_RR50_map), 0))
    Model_1_expected_placebo_MPC  = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_MPC_map),  0))
    Model_1_expected_placebo_TTP  = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_TTP_map),  0))
    Model_2_expected_placebo_RR50 = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_RR50_map), 0))
    Model_2_expected_placebo_MPC  = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_MPC_map),  0))
    Model_2_expected_placebo_TTP  = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_TTP_map),  0))
    Model_1_expected_drug_RR50    = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_RR50_map), 0))
    Model_1_expected_drug_MPC     = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_MPC_map),  0))
    Model_1_expected_drug_TTP     = np.sum(np.nansum(np.multiply(H_model_1, expected_placebo_TTP_map),  0))
    Model_2_expected_drug_RR50    = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_RR50_map), 0))
    Model_2_expected_drug_MPC     = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_MPC_map),  0))
    Model_2_expected_drug_TTP     = np.sum(np.nansum(np.multiply(H_model_2, expected_placebo_TTP_map),  0))
    Model_1_RR50_stat_power       = np.sum(np.nansum(np.multiply(H_model_1, RR50_stat_power_map), 0))
    Model_1_MPC_stat_power        = np.sum(np.nansum(np.multiply(H_model_1, MPC_stat_power_map),  0))
    Model_1_TTP_stat_power        = np.sum(np.nansum(np.multiply(H_model_1, TTP_stat_power_map),  0))
    Model_2_RR50_stat_power       = np.sum(np.nansum(np.multiply(H_model_2, RR50_stat_power_map), 0))
    Model_2_MPC_stat_power        = np.sum(np.nansum(np.multiply(H_model_2, MPC_stat_power_map),  0))
    Model_2_TTP_stat_power        = np.sum(np.nansum(np.multiply(H_model_2, TTP_stat_power_map),  0))
    Model_1_RR50_type_1_error     = np.sum(np.nansum(np.multiply(H_model_1, RR50_type_1_error_map), 0))
    Model_1_MPC_type_1_error      = np.sum(np.nansum(np.multiply(H_model_1, MPC_type_1_error_map),  0))
    Model_1_TTP_type_1_error      = np.sum(np.nansum(np.multiply(H_model_1, TTP_type_1_error_map),  0))
    Model_2_RR50_type_1_error     = np.sum(np.nansum(np.multiply(H_model_2, RR50_type_1_error_map), 0))
    Model_2_MPC_type_1_error      = np.sum(np.nansum(np.multiply(H_model_2, MPC_type_1_error_map),  0))
    Model_2_TTP_type_1_error      = np.sum(np.nansum(np.multiply(H_model_2, TTP_type_1_error_map),  0))

    return [Model_1_expected_placebo_RR50, Model_1_expected_placebo_MPC, Model_1_expected_placebo_TTP,
            Model_2_expected_placebo_RR50, Model_2_expected_placebo_MPC, Model_2_expected_placebo_TTP,
            Model_1_expected_drug_RR50,    Model_1_expected_drug_MPC,    Model_1_expected_drug_TTP, 
            Model_2_expected_drug_RR50,    Model_2_expected_drug_MPC,    Model_2_expected_drug_TTP,
            Model_1_RR50_stat_power,       Model_1_MPC_stat_power,       Model_1_TTP_stat_power,
            Model_2_RR50_stat_power,       Model_2_MPC_stat_power,       Model_2_TTP_stat_power,
            Model_1_RR50_type_1_error,     Model_1_MPC_type_1_error,     Model_1_TTP_type_1_error,
            Model_2_RR50_type_1_error,     Model_2_MPC_type_1_error,     Model_2_TTP_type_1_error,
            expected_placebo_RR50_map,     expected_placebo_MPC_map,     expected_placebo_TTP_map,
            expected_drug_RR50_map,        expected_drug_MPC_map,        expected_drug_TTP_map,
            RR50_stat_power_map,           MPC_stat_power_map,           TTP_stat_power_map,
            RR50_type_1_error_map,         MPC_type_1_error_map,         TTP_type_1_error_map,
            H_model_1,                     H_model_2                 ]


def store_map(data_map, 
              data_map_file_name, data_map_meta_data_file_name,
              x_axis_start, x_axis_stop, x_axis_step,
              y_axis_start, y_axis_stop, y_axis_step):
    '''

    This function takes a data map as the parameters used to create its axes (referred to as the metadata)

    and puts all the information into JSON files which are located in the same folder as this script.

    Inputs:

        1) data_map:

            (2D Numpy array) - a 2D numpy array containing data to be stored now in a JSON
                               
                               file and plotted later
        
        2) data_map_file_name:

            (string) - the name of the JSON file which will contain the data map to be plotted

                       later
        
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
    
    Outputs:

        Technically None

    '''

    # get the file path for the JSON file that will store the data map 
    data_map_file_path = os.getcwd() + '/' + data_map_file_name + '.json'

    # get the file path for the JSON file that will store the data map metadata
    data_map_metadata_file_path = os.getcwd() + '/' + data_map_meta_data_file_name + '.json'

    # put all of the metadata into one array
    metadata = np.array([x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step])

    # open/create the JSON file that will store the data map 
    with open(data_map_file_path, 'w+') as map_storage_file:

        # store the data map
        json.dump(data_map.tolist(), map_storage_file)

    # open/create the JSON file that will store the data map 
    with open(data_map_metadata_file_path, 'w+') as map_metadata_storage_file:

        # store the data map
        json.dump(metadata.tolist(), map_metadata_storage_file)


def save_NV_model_endpoint_statistics(Model_1_expected_placebo_RR50, Model_1_expected_placebo_MPC, Model_1_expected_placebo_TTP,
                                      Model_2_expected_placebo_RR50, Model_2_expected_placebo_MPC, Model_2_expected_placebo_TTP,
                                      Model_1_expected_drug_RR50,    Model_1_expected_drug_MPC,    Model_1_expected_drug_TTP, 
                                      Model_2_expected_drug_RR50,    Model_2_expected_drug_MPC,    Model_2_expected_drug_TTP,
                                      Model_1_RR50_stat_power,       Model_1_MPC_stat_power,       Model_1_TTP_stat_power,
                                      Model_2_RR50_stat_power,       Model_2_MPC_stat_power,       Model_2_TTP_stat_power,
                                      Model_1_RR50_type_1_error,     Model_1_MPC_type_1_error,     Model_1_TTP_type_1_error,
                                      Model_2_RR50_type_1_error,     Model_2_MPC_type_1_error,     Model_2_TTP_type_1_error,
                                      NV_model_endpoint_statistics_text_file_name):
    '''

    This function stores the expected placebo responses for NV model 1 ande NV model 2 as predicted by the multiplication of the 
    
    SNR maps and the histograms for both NV models into a text file. 

    Inputs:

        1) Model_1_expected_placebo_RR50:
            
            (float) - the collective expected placebo arm 50% responder rate for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the 50% responder rate expected placebo response maps
        
        2) Model_1_expected_placebo_MPC:
                    
            (float) - the collective expected placebo arm median percent change for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the median percent change expected placebo response maps
        
        3) Model_1_expected_placebo_TTP:
                            
            (float) - the collective expected placebo arm time-to-prerandomization for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the time-to-prerandomization expected placebo response maps
        
        4) Model_2_expected_placebo_RR50:
            
            (float) - the collective expected placebo arm 50% responder rate for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the 50% responder rate expected placebo response maps
        
        5) Model_2_expected_placebo_MPC:
                            
            (float) - the collective expected placebo arm median percent change for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the median percent change expected placebo response maps
        
        6) Model_2_expected_placebo_TTP:
                            
            (float) - the collective expected placebo arm time-to-prerandomization for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the time-to-prerandomization expected placebo response maps
        
        7) Model_1_expected_drug_RR50:
            
            (float) - the collective expected drug arm 50% responder rate for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the 50% responder rate expected drug response maps
        
        8) Model_1_expected_drug_MPC:
            
            (float) - the collective expected drug arm median percent change for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the median percent change expected drug response maps
        
        9) Model_1_expected_drug_TTP:
                            
            (float) - the collective expected drug arm time-to-prerandomization for all of the patients from model 1, determined by a 
            
                      multiplication of the Model 1 patient histogram and the time-to-prerandomization expected placebo response maps
        
        10) Model_2_expected_drug_RR50:
            
            (float) - the collective expected drug arm 50% responder rate for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the 50% responder rate expected drug response maps
        
        11) Model_2_expected_drug_MPC:
            
            (float) - the collective expected drug arm median percent change for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the median percent change expected drug response maps
        
        12) Model_2_expected_drug_TTP:
                            
            (float) - the collective expected drug arm time-to-prerandomization for all of the patients from model 2, determined by a 
            
                      multiplication of the Model 2 patient histogram and the time-to-prerandomization expected placebo response maps
        
        1) Model_1_RR50_stat_power:
            
            (float) - the collective statistical power for all of the patients from model 1, determined by a multiplication of the 
            
                      Model 1 patient histogram and the 50% responder rate statistical power maps
        
        2) Model_1_MPC_stat_power:
        
            (float) - the collective statistical power for all of the patients from model 1, determined by a multiplication of the 
                      
                      Model 1 patient histogram and the median percent change statistical power maps
        
        3) Model_1_TTP_stat_power:
        
            (float) - the collective statistical power for all of the patients from model 1, determined by a multiplication of the 
                      
                      Model 1 patient histogram and the time-to-prerandomization statistical power maps
        
        4)  Model_2_RR50_stat_power:
        
            (float) - the collective statistical power for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the 50% responder rate statistical power maps

        5) Model_2_MPC_stat_power:
        
            (float) - the collective statistical power for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the median percent change statistical power maps
        
        6) Model_2_TTP_stat_power:
        
            (float) - the collective statistical power for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the time-to-prerandomization statistical power maps
        
        Model_1_RR50_type_1_error:
            
            (float) - the collective type-1 error for all of the patients from model 1, determined by a multiplication of the 
            
                      Model 1 patient histogram and the 50% responder rate statistical power maps
        
        Model_1_MPC_type_1_error:
        
            (float) - the collective type-1 error for all of the patients from model 1, determined by a multiplication of the 
                      
                      Model 1 patient histogram and the median percent change type-1 error maps
        
        Model_1_TTP_type_1_error:
        
            (float) - the collective type-1 error for all of the patients from model 1, determined by a multiplication of the 
                      
                      Model 1 patient histogram and the time-to-prerandomization type-1 error maps
        
        Model_2_RR50_type_1_error:
        
            (float) - the collective type-1 error for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the 50% responder rate type-1 error maps

        Model_2_MPC_type_1_error:
        
            (float) - the collective type-1 error for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the median percent change type-1 error maps
        
        Model_2_TTP_type_1_error:
                
            (float) - the collective type-1 error for all of the patients from model 2, determined by a multiplication of the 
            
                      Model 2 patient histogram and the time-to-prerandomization type-1 error maps
        
        7) NV_model_statistical_power_text_file_name:

            (string) - the name of the text file which will contain the endpoint statistics for NV model 1 and NV model 2

    Outputs:

        Technically None

    '''

    # open/create the text file that will contain all the enpdoint statistics
    with open(os.getcwd() + '/' + NV_model_endpoint_statistics_text_file_name + '.txt', 'w+') as text_file:

        # format all the endpoint statistics having to do with the placebo arm
        placebo_response =  '\nModel 1 expected placebo RR50: ' + str(Model_1_expected_placebo_RR50) + \
                            '\nModel 1 expected placebo MPC: '  + str(Model_1_expected_placebo_MPC)  + \
                            '\nModel 1 expected placebo TTP: '  + str(Model_1_expected_placebo_TTP)  + \
                            '\nModel 2 expected placebo RR50: ' + str(Model_2_expected_placebo_RR50) + \
                            '\nModel 2 expected placebo MPC: '  + str(Model_2_expected_placebo_MPC)  + \
                            '\nModel 2 expected placebo TTP: '  + str(Model_2_expected_placebo_TTP)
        
        # format all the endpoint statistics having to do with the drug arm
        drug_response =  '\nModel 1 expected drug RR50: ' + str(Model_1_expected_drug_RR50) + \
                         '\nModel 1 expected drug MPC: '  + str(Model_1_expected_drug_MPC)  + \
                         '\nModel 1 expected drug TTP: '  + str(Model_1_expected_drug_TTP)  + \
                         '\nModel 2 expected drug RR50: ' + str(Model_2_expected_drug_RR50) + \
                         '\nModel 2 expected drug MPC: '  + str(Model_2_expected_drug_MPC)  + \
                         '\nModel 2 expected drug TTP: '  + str(Model_2_expected_drug_TTP)
        
        # format all the endpoint statistics having to do with the statistical power
        stat_power =  '\nModel 1 RR50 statistical power: ' + str(Model_1_RR50_type_1_error) + \
                      '\nModel 1 MPC statistical power: '  + str(Model_1_MPC_type_1_error)  + \
                      '\nModel 1 TTP statistical power: '  + str(Model_1_TTP_type_1_error)  + \
                      '\nModel 2 RR50 statistical power: ' + str(Model_2_RR50_type_1_error) + \
                      '\nModel 2 MPC statistical power: '  + str(Model_2_MPC_type_1_error)  + \
                      '\nModel 2 TTP statistical power: '  + str(Model_2_TTP_type_1_error)
        
        # format all the endpoint statistics having to do with the type-1 error
        type_1_error =  '\nModel 1 RR50 type-1 error: ' + str(Model_1_RR50_stat_power) + \
                        '\nModel 1 MPC type-1 error: '  + str(Model_1_MPC_stat_power)  + \
                        '\nModel 1 TTP type-1 error: '  + str(Model_1_TTP_stat_power)  + \
                        '\nnModel 2 RR50 type-1 error: ' + str(Model_2_RR50_stat_power) + \
                        '\nModel 2 MPC type-1 error: '  + str(Model_2_MPC_stat_power)  + \
                        '\nModel 2 TTP type-1 error: '  + str(Model_2_TTP_stat_power)
        
        # put all the data together into one string
        data = '\n\n' + placebo_response + '\n\n' + drug_response + '\n\n' + stat_power + '\n\n' + type_1_error + '\n\n'

        # write the string full of data to the designated text file
        text_file.write(data)


def main(shape_1, scale_1, alpha_1, beta_1, 
         shape_2, scale_2, alpha_2, beta_2,
         num_patients_per_model, num_months_per_patient,
         start_monthly_mean, stop_monthly_mean, step_monthly_mean, 
         start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
         num_baseline_months, num_testing_months, min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
         placebo_mu, placebo_sigma,  drug_mu, drug_sigma,
         expected_placebo_RR50_file_name, expected_placebo_RR50_metadata_file_name,
         expected_placebo_MPC_file_name,  expected_placebo_MPC_metadata_file_name,
         expected_placebo_TTP_file_name,  expected_placebo_TTP_metadata_file_name,
         expected_drug_RR50_file_name,    expected_drug_RR50_metadata_file_name,
         expected_drug_MPC_file_name,     expected_drug_MPC_metadata_file_name,
         expected_drug_TTP_file_name,     expected_drug_TTP_metadata_file_name,
         RR50_stat_power_file_name,       RR50_stat_power_metadata_file_name,
         MPC_stat_power_file_name,        MPC_stat_power_metadata_file_name,
         TTP_stat_power_file_name,        TTP_stat_power_metadata_file_name,
         RR50_type_1_error_file_name,     RR50_type_1_error_metadata_file_name,
         MPC_type_1_error_file_name,      MPC_type_1_error_metadata_file_name,
         TTP_type_1_error_file_name,      TTP_type_1_error_metadata_file_name,
         H_model_1_file_name, H_model_1_metadata_file_name,
         H_model_2_file_name, H_model_2_metadata_file_name,
         NV_model_endpoint_statistics_text_file_name):
    '''

    This function is the main function that should be called in order to fulfill this script's primary purpose,

    which is generating and storing the data needed to plot the expected placebo response maps to be plotted later.

    Inputs:

        1) shape_1:
        
            (float) - the first parameter for NV model 1
        
        2) scale_1:
        
            (float) - the second parameter for NV model 1
        
        3) alpha_1:
        
            (float) - the third parameter for NV model 1
        
        4) beta_1:
        
            (float) - the fourth parameter for NV model 1

        5) shape_2:
        
            (float) - the first parameter for NV model 2
        
        6) scale_2:
        
            (float) - the second parameter for NV model 2

        7) alpha_2:
        
            (float) - the third parameter for NV model 2
        
        8) beta_2:

            (float) -  the fourth parameter for NV model 2

        9) num_patients_per_model:
                
            (int) - the number of patients to generate for the histograms of models 1 and 2
        
        10) num_months_per_patient:

            (int) - the number of months that each patient should have in both NV models

        11) start_monthly_mean:
        
            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps=
        
        12) stop_monthly_mean:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps
        
        13) step_monthly_mean:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps

        14) start_monthly_std_dev:
                
            (float) - the beginning of the monthly seizure count mean axis for all expected endpoint maps
        
        15) stop_monthly_std_dev:

            (float) - the end of the monthly seizure count mean axis for all expected endpoint maps
        
        16) step_monthly_std_dev:

            (float) - the spaces in between each location on the monthly seizure count mean axis for all expected endpoint maps

        17) num_baseline_months:
                        
            (int) - the number of baseline months in each patient's seizure diary
        
        18) num_testing_months:
                
            (int) - the number of testing months in each patient's seizure diary
        
        19) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        20) num_patients_per_trial_arm:

            (int) - the number of patients generated per trial arm
        
        21) num_trials:

            (int) -  the number of trials used to estimate the expected endpoints at each point in the expected placebo response maps
        
        22) placebo_mu:

            (float) - the mean of the placebo effect

        23) placebo_sigma:
        
            (float) - the standard deviation of the placebo effect

        24) drug_mu:

            (float) - the mean of the drug effect

        25) drug_sigma:

            (float) - the standard deviation of the drug effect

        26) expected_RR50_file_name:

            (string) - the name of the JSON file which will contain the expected 50% responder rate placebo response 
            
                       map to be plotted later
        
        27) expected_RR50_metadata_file_name:

            (string) - the name of the JSON file which will contain the expected 50% responder rate placebo response 
            
                       map metadata to be plotted later

        28) expected_MPC_file_name:

            (string) - the name of the JSON file which will contain the expected median percent change placebo response 
            
                       map to be plotted later
        
        29) expected_MPC_metadata_file_name:

            (string) - the name of the JSON file which will contain the expected median percent change placebo response 
            
                       map metadata to be plotted later
        
        30) expected_TTP_file_name: 
        
            (string) - the name of the JSON file which will contain the expected time-to-prerandomization placebo response 
            
                       map to be plotted later

        31) expected_TTP_metadata_file_name:
        
            (string) - the name of the JSON file which will contain the expected time-to-prerandomization placebo response 
            
                       map metadata to be plotted later
        
        32) H_model_1_file_name:
        
            (string) - the name of the JSON file which will contain the histogram of NV model 1 to be plotted later

        33) H_model_1_metadata_file_name:
                
            (string) - the name of the JSON file which will contain the histogram of NV model 1 metadata to be plotted 
            
                       later

        34) H_model_2_file_name:
                
            (string) - the name of the JSON file which will contain the histogram of NV model 2 to be plotted later

        35) H_model_2_metadata_file_name:
        
            (string) - the name of the JSON file which will contain the histogram of NV model 2 metadata to be plotted 
            
                       later

        36) NV_model_endpoint_statistics_text_file_name:

            (string) - the name of the text file which will contain the endpoint statistics for NV 
                       
                       model 1 and NV model 2
    
    Outputs:

        Technically None

    '''

    # generate all of the expected endpoint maps
    [Model_1_expected_placebo_RR50, Model_1_expected_placebo_MPC, Model_1_expected_placebo_TTP,
     Model_2_expected_placebo_RR50, Model_2_expected_placebo_MPC, Model_2_expected_placebo_TTP,
     Model_1_expected_drug_RR50,    Model_1_expected_drug_MPC,    Model_1_expected_drug_TTP, 
     Model_2_expected_drug_RR50,    Model_2_expected_drug_MPC,    Model_2_expected_drug_TTP,
     Model_1_RR50_stat_power,       Model_1_MPC_stat_power,       Model_1_TTP_stat_power,
     Model_2_RR50_stat_power,       Model_2_MPC_stat_power,       Model_2_TTP_stat_power,
     Model_1_RR50_type_1_error,     Model_1_MPC_type_1_error,     Model_1_TTP_type_1_error,
     Model_2_RR50_type_1_error,     Model_2_MPC_type_1_error,     Model_2_TTP_type_1_error,
     expected_placebo_RR50_map,     expected_placebo_MPC_map,     expected_placebo_TTP_map,
     expected_drug_RR50_map,        expected_drug_MPC_map,        expected_drug_TTP_map,
     RR50_stat_power_map,           MPC_stat_power_map,           TTP_stat_power_map,
     RR50_type_1_error_map,         MPC_type_1_error_map,         TTP_type_1_error_map,
     H_model_1,                     H_model_2                 ] = \
        generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, 
                          shape_2, scale_2, alpha_2, beta_2, 
                          num_patients_per_model, num_months_per_patient,
                          start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
                          start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
                          num_patients_per_trial_arm, num_trials,
                          num_baseline_months, num_testing_months, min_req_base_sz_count,
                          placebo_mu, placebo_sigma,  drug_mu, drug_sigma)

    # store the expected 50% responder rate placebo arm response map
    store_map(expected_placebo_RR50_map, 
              expected_placebo_RR50_file_name, expected_placebo_RR50_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the expected median percent change placebo arm response map
    store_map(expected_placebo_MPC_map, 
              expected_placebo_MPC_file_name, expected_placebo_MPC_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the expected time-to-prerandomization placebo arm response map
    store_map(expected_placebo_TTP_map, 
              expected_placebo_TTP_file_name, expected_placebo_TTP_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the expected 50% responder rate drug arm response map
    store_map(expected_drug_RR50_map, 
              expected_drug_RR50_file_name, expected_drug_RR50_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the expected median percent change drug arm response map
    store_map(expected_drug_MPC_map, 
              expected_drug_MPC_file_name, expected_drug_MPC_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the expected time-to-prerandomization drug arm response map
    store_map(expected_drug_TTP_map, 
              expected_drug_TTP_file_name, expected_drug_TTP_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the 50% responder rate statistical power map
    store_map(RR50_stat_power_map, 
              RR50_stat_power_file_name, RR50_stat_power_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the median percent change statistical power map
    store_map(MPC_stat_power_map, 
              MPC_stat_power_file_name, MPC_stat_power_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the time-to-prerandomization statistical power map
    store_map(TTP_stat_power_map, 
              TTP_stat_power_file_name, TTP_stat_power_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the 50% responder rate type-1 error map
    store_map(RR50_type_1_error_map, 
              RR50_type_1_error_file_name, RR50_type_1_error_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the median percent change type-1 error map
    store_map(MPC_type_1_error_map, 
              MPC_type_1_error_file_name, MPC_type_1_error_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the time-to-prerandomization type-1 error map
    store_map(TTP_type_1_error_map, 
              TTP_type_1_error_file_name, TTP_type_1_error_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)

    # store the histogram of Model 1
    store_map(H_model_1,
              H_model_1_file_name, H_model_1_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the histogram of Model 1
    store_map(H_model_2,
              H_model_2_file_name, H_model_2_metadata_file_name,
              start_monthly_mean,    stop_monthly_mean,    step_monthly_mean, 
              start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev)
    
    # store the estimated placebo responses for NV model 1 and NV model 2
    save_NV_model_endpoint_statistics(Model_1_expected_placebo_RR50, Model_1_expected_placebo_MPC, Model_1_expected_placebo_TTP,
                                      Model_2_expected_placebo_RR50, Model_2_expected_placebo_MPC, Model_2_expected_placebo_TTP,
                                      Model_1_expected_drug_RR50,    Model_1_expected_drug_MPC,    Model_1_expected_drug_TTP, 
                                      Model_2_expected_drug_RR50,    Model_2_expected_drug_MPC,    Model_2_expected_drug_TTP,
                                      Model_1_RR50_stat_power,       Model_1_MPC_stat_power,       Model_1_TTP_stat_power,
                                      Model_2_RR50_stat_power,       Model_2_MPC_stat_power,       Model_2_TTP_stat_power,
                                      Model_1_RR50_type_1_error,     Model_1_MPC_type_1_error,     Model_1_TTP_type_1_error,
                                      Model_2_RR50_type_1_error,     Model_2_MPC_type_1_error,     Model_2_TTP_type_1_error,
                                      NV_model_endpoint_statistics_text_file_name)


if(__name__=='__main__'):

    start_time_in_seconds = time.time()

    # define the parameters for NV model 1 and NV model 2
    shape_1 = 24.143
    scale_1 = 297.366
    alpha_1 = 284.024
    beta_1 = 369.628
    shape_2 = 111.313
    scale_2 = 296.728
    alpha_2 = 296.339
    beta_2 = 243.719

    # take in the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('array', nargs='+')
    args = parser.parse_args()
    arg_array = args.array

    # obtain the information needed for the x-axis of the expected placebo response maps
    start_monthly_mean = float(arg_array[0])
    stop_monthly_mean = float(arg_array[1])
    step_monthly_mean = float(arg_array[2])

    # obtain the information needed for the y-axis of the expected placebo response maps
    start_monthly_std_dev = float(arg_array[3])
    stop_monthly_std_dev = float(arg_array[4])
    step_monthly_std_dev = float(arg_array[5])

    # obtain the parameters for estimating the placebo response at each point on the expected placebo response maps
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

    # obtain the parameters needed for generating the histograms of the model 1 and model 2 patients
    num_patients_per_model = int(arg_array[15])
    num_months_per_patient = int(arg_array[16])

    # set the names of the files that the data/metadata of the endpoint statistic maps will be stored in
    expected_placebo_RR50_file_name           = 'expected_placebo_RR50_power_map'
    expected_placebo_RR50_metadata_file_name  = 'expected_placebo_RR50_map_metadata'
    expected_placebo_MPC_file_name            = 'expected_placebo_MPC_power_map'
    expected_placebo_MPC_metadata_file_name   = 'expected_placebo_MPC_map_metadata'
    expected_placebo_TTP_file_name            = 'expected_placebo_TTP_power_map'
    expected_placebo_TTP_metadata_file_name   = 'expected_placebo_TTP_map_metadata'
    expected_drug_RR50_file_name              = 'expected_drug_RR50_power_map'
    expected_drug_RR50_metadata_file_name     = 'expected_drug_RR50_map_metadata'
    expected_drug_MPC_file_name               = 'expected_drug_MPC_power_map'
    expected_drug_MPC_metadata_file_name      = 'expected_drug_MPC_map_metadata'
    expected_drug_TTP_file_name               = 'expected_drug_TTP_power_map'
    expected_drug_TTP_metadata_file_name      = 'expected_drug_TTP_map_metadata'
    RR50_stat_power_file_name                 = 'RR50_stat_power_map'
    RR50_stat_power_metadata_file_name        = 'RR50_stat_power_map_metadata'
    MPC_stat_power_file_name                  = 'MPC_stat_power_map'
    MPC_stat_power_metadata_file_name         = 'MPC_stat_power_map_metadata'
    TTP_stat_power_file_name                  = 'TTP_stat_power_map'
    TTP_stat_power_metadata_file_name         = 'TTP_stat_power_map_metadata'
    RR50_type_1_error_file_name               = 'RR50_stat_power_map'
    RR50_type_1_error_metadata_file_name      = 'RR50_stat_power_map_metadata'
    MPC_type_1_error_file_name                = 'MPC_stat_power_map'
    MPC_type_1_error_metadata_file_name       = 'MPC_stat_power_map_metadata'
    TTP_type_1_error_file_name                = 'TTP_stat_power_map'
    TTP_type_1_error_metadata_file_name       = 'TTP_stat_power_map_metadata'
    H_model_1_file_name                       = 'H_model_1_hist'
    H_model_1_metadata_file_name              = 'H_model_1_hist_metadata'
    H_model_2_file_name                       = 'H_model_2_hist'
    H_model_2_metadata_file_name              = 'H_model_2_hist_metadata'

    # obtain the name of text file which will contain the placebo responses for NV model 1 and NV model 2
    NV_model_endpoint_statistics_text_file_name = 'NV_model_endpoint_statistics'

    # call the main() function
    main(shape_1, scale_1, alpha_1, beta_1, 
         shape_2, scale_2, alpha_2, beta_2,
         num_patients_per_model, num_months_per_patient,
         start_monthly_mean, stop_monthly_mean, step_monthly_mean, 
         start_monthly_std_dev, stop_monthly_std_dev, step_monthly_std_dev,
         num_baseline_months, num_testing_months, min_req_base_sz_count, num_patients_per_trial_arm, num_trials,
         placebo_mu, placebo_sigma,  drug_mu, drug_sigma,
         expected_placebo_RR50_file_name, expected_placebo_RR50_metadata_file_name,
         expected_placebo_MPC_file_name,  expected_placebo_MPC_metadata_file_name,
         expected_placebo_TTP_file_name,  expected_placebo_TTP_metadata_file_name,
         expected_drug_RR50_file_name,    expected_drug_RR50_metadata_file_name,
         expected_drug_MPC_file_name,     expected_drug_MPC_metadata_file_name,
         expected_drug_TTP_file_name,     expected_drug_TTP_metadata_file_name,
         RR50_stat_power_file_name,       RR50_stat_power_metadata_file_name,
         MPC_stat_power_file_name,        MPC_stat_power_metadata_file_name,
         TTP_stat_power_file_name,        TTP_stat_power_metadata_file_name,
         RR50_type_1_error_file_name,     RR50_type_1_error_metadata_file_name,
         MPC_type_1_error_file_name,      MPC_type_1_error_metadata_file_name,
         TTP_type_1_error_file_name,      TTP_type_1_error_metadata_file_name,
         H_model_1_file_name, H_model_1_metadata_file_name,
         H_model_2_file_name, H_model_2_metadata_file_name,
         NV_model_endpoint_statistics_text_file_name)
    
    stop_time_in_seconds = time.time()

    total_time_in_minutes  = (stop_time_in_seconds - start_time_in_seconds)/60

    svem = psutil.virtual_memory()
    total_mem_in_bytes = svem.total
    available_mem_in_bytes = svem.available
    used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
    used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)

    print('\n\ncpu time in minutes: ' + str(total_time_in_minutes) + '\nmemory usage in GB: ' + str(used_mem_in_gigabytes) + '\n' )


