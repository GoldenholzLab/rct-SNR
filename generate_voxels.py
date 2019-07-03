import numpy as np
from scipy import stats
from lifelines.statistics import logrank_test
import time


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


def generate_one_trial_population(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                                  rct_params_monthly_scale, effect_params):
    '''

    Purpose:

        This function generates the patient diaries needed for one trial which is assumed to have a

        drug arm and a placebo arm. The patient populations of both the drug arm and the placebo arm

        both have the same number of patients.

    Inputs:

        1) monthly_mean:

            (float) - the mean of the daily seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the daily seizure counts in each patient's seizure diary
        
        3) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        4) rct_params_monthly_scale:

            (1D Numpy array) - a numpy array containing the RCT design parameters for the trial to be generated

                               on a monthly time scale. This Numpy array contains the folllowing quantities:

                                    4.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    4.b) num_baseline_months:

                                        (int) - the number of baseline months in each patient's seizure diary

                                    4.c) num_testing_months:

                                        (int) - the number of testing months in each patient's seizure diary

        5) effect_params:

            (1D Numpy array) - a numpy array containing the statistical parameters needed to generate the placebo
                               
                               and drug effects according to the implemented algorithms. Ths numpy array contains

                               the following quantities:

                                    5.a) placebo_mu:

                                        (float) - the mean of the placebo effect

                                    5.b) placebo_sigma:
        
                                        (float) - the standard deviation of the placebo effect

                                    5.c) drug_mu:

                                        (float) - the mean of the drug effect

                                    5.d) drug_sigma:

                                        (float) - the standard deviation of the drug effect

    Outputs:

        1) placebo_arm_daily_seizure_diaries:
        
            (2D Numpy array) - an array of the patient diaries from the placebo arm of one trial

        2) drug_arm_daily_seizure_diaries:

            (2D Numpy array) - an array of the patient diaries from the drug arm of one trial

    '''

    # extract the RCT design parameters
    num_patients_per_trial_arm = rct_params_monthly_scale[0]
    num_baseline_months        = rct_params_monthly_scale[1]
    num_testing_months         = rct_params_monthly_scale[2]

    # extract the statistical parameters for the drug and placebo effect
    placebo_mu    = effect_params[0]
    placebo_sigma = effect_params[1]
    drug_mu       = effect_params[2]
    drug_sigma    = effect_params[3]
    
    # convert monthly parameters into daily parameters
    num_baseline_days = num_baseline_months*28
    num_testing_days  = num_testing_months*28
    daily_mean        = monthly_mean/28
    daily_std_dev     = monthly_std_dev/np.sqrt(28)

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


def generate_voxel_populations(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                               rct_params_monthly_scale, effect_params):
    '''

    Purpose:

        TO-DO

    Inputs:

        1) monthly_mean:

            (float) - the mean of the daily seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the daily seizure counts in each patient's seizure diary
        
        3) min_req_base_sz_count:

            (int) - the minimum number of required baseline seizure counts
        
        4) rct_params_monthly_scale:

            (1D Numpy array) - a numpy array containing the RCT design parameters for the trial to be generated

                               on a monthly time scale. This Numpy array contains the folllowing quantities:

                                    4.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    4.b) num_baseline_months:

                                        (int) - the number of baseline months in each patient's seizure diary

                                    4.c) num_testing_months:

                                        (int) - the number of testing months in each patient's seizure diary

        5) effect_params:

            (1D Numpy array) - a numpy array containing the statistical parameters needed to generate the placebo
                               
                               and drug effects according to the implemented algorithms. Ths numpy array contains

                               the following quantities:

                                    5.a) placebo_mu:

                                        (float) - the mean of the placebo effect

                                    5.b) placebo_sigma:
        
                                        (float) - the standard deviation of the placebo effect

                                    5.c) drug_mu:

                                        (float) - the mean of the drug effect

                                    5.d) drug_sigma:

                                        (float) - the standard deviation of the drug effect

    Outputs:

        1) placebo_arm_daily_seizure_diaries
        
            (2D Numpy array) - 

        2) drug_arm_daily_seizure_diaries:

            (2D Numpy array) - 
        
        3) first_type_1_arm_daily_seizure_diaries

            (2D Numpy array) - 
        
        4) second_type_1_arm_daily_seizure_diaries

            (2D Numpy array) - 

    '''
    # make sure the placebo and drug effect parameters are clearly labeled
    effect_params_placebo_vs_drug = effect_params.copy()

    # make the second trial such that there is no drug effect for the sake of testing type-1 error
    effect_params_placebo_vs_placebo    = np.zeros(4)
    effect_params_placebo_vs_placebo[0] = effect_params[0]
    effect_params_placebo_vs_placebo[1] = effect_params[1]

    # generate one set of daily seizure diaries for each placebo arm and drug arm of a trial (two sets in total)
    [placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries] = \
        generate_one_trial_population(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                                      rct_params_monthly_scale, effect_params_placebo_vs_drug)
                
    # generate two sets of daily seizure diaries, both from a placebo arm (meant for calculating type 1 Error)
    [first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries] = \
        generate_one_trial_population(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                                      rct_params_monthly_scale, effect_params_placebo_vs_placebo)
    
    return [placebo_arm_daily_seizure_diaries,      drug_arm_daily_seizure_diaries,
            first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries]


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


def calculate_trial_endpoints(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                              num_patients_per_trial_arm, num_months_baseline, num_months_testing):
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
    num_baseline_days = 28*num_months_baseline
    num_testing_days = 28*num_months_testing

    # calculate the percent changes and times-to-prerandomization for each patient in boht the placebo and drug arm groups
    placebo_percent_changes = calculate_percent_changes(placebo_arm_daily_seizure_diaries, num_baseline_days, num_patients_per_trial_arm)
    drug_percent_changes = calculate_percent_changes(drug_arm_daily_seizure_diaries, num_baseline_days, num_patients_per_trial_arm)
    placebo_TTP_times = calculate_times_to_prerandomization(placebo_arm_daily_seizure_diaries, num_months_baseline, num_testing_days, num_patients_per_trial_arm)
    drug_TTP_times = calculate_times_to_prerandomization(drug_arm_daily_seizure_diaries, num_months_baseline, num_testing_days, num_patients_per_trial_arm)

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


def calculate_voxel_endpoints(monthly_mean, monthly_std_dev, min_req_base_sz_count,
                              rct_params_monthly_scale, effect_params):

    if(monthly_mean != 0):

        if(monthly_std_dev > np.sqrt(monthly_mean)):
        
            num_months_baseline = rct_params_monthly_scale[1]
            num_months_testing  = rct_params_monthly_scale[2]

            [placebo_arm_daily_seizure_diaries,      drug_arm_daily_seizure_diaries,
             first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries] = \
                generate_voxel_populations(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                                           rct_params_monthly_scale, effect_params)

            trial_endpoints = \
                calculate_trial_endpoints(placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries,
                                          num_patients_per_trial_arm, num_months_baseline, num_months_testing)

            pvp_trial_endpoints = \
                calculate_trial_endpoints(first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries,
                                          num_patients_per_trial_arm, num_months_baseline, num_months_testing)
        
        else:

            trial_endpoints = np.array([np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan]),
            pvp_trial_endpoints = np.array([np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan])
    
    else:

        trial_endpoints = np.array([np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan]),
        pvp_trial_endpoints = np.array([np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan])
    
    return [trial_endpoints, pvp_trial_endpoints]


if(__name__ == '__main__'):

    start_monthly_mean        = 0
    stop_monthly_mean         = 4
    step_monthly_mean         = 0.1
    start_monthly_std_dev     = 0
    stop_monthly_std_dev      = 5
    step_monthly_std_dev      = 0.1
    max_min_req_base_sz_count = 5

    num_patients_per_trial_arm = 153
    num_months_baseline        = 2
    num_months_testing         = 3
    rct_params_monthly_scale   = np.array([num_patients_per_trial_arm, num_months_baseline, num_months_testing])

    placebo_mu = 0
    placebo_sigma = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05
    effect_params = np.array([placebo_mu, placebo_sigma, drug_mu, drug_sigma])

    num_trials = 3

    #-------------------------------------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------------#

    import os
    import pandas as pd
    runtimes_in_seconds = np.zeros(num_trials)

    for min_req_base_sz_count in range(max_min_req_base_sz_count + 1):

        monthly_mean_axis        = np.arange(start_monthly_mean, stop_monthly_mean + step_monthly_mean, step_monthly_mean)
        monthly_std_dev_axis     = np.arange(start_monthly_std_dev, stop_monthly_std_dev + step_monthly_std_dev, step_monthly_std_dev)
        monthly_mean_axis_len    = len(monthly_mean_axis)
        monthly_std_dev_axis_len = len(monthly_std_dev_axis)
        runtime_matrix = np.zeros((monthly_std_dev_axis_len, monthly_mean_axis_len))

        for monthly_mean_index in range(monthly_mean_axis_len):

            for monthly_std_dev_index in range(monthly_std_dev_axis_len):

                monthly_mean = monthly_mean_axis[monthly_mean_index]
                monthly_std_dev = monthly_std_dev_axis[monthly_std_dev_index]

                for trial_num in range(num_trials):

                    monthly_mean = monthly_mean_axis[monthly_mean_index]
                    monthly_std_dev = monthly_std_dev_axis[monthly_std_dev_index]

                    start_time_in_seconds = time.time()

                    [trial_endpoints, pvp_trial_endpoints] = \
                        calculate_voxel_endpoints(monthly_mean, monthly_std_dev, min_req_base_sz_count,
                                                  rct_params_monthly_scale, effect_params)
        
                    stop_time_in_seconds = time.time()
                    runtime_in_seconds = stop_time_in_seconds - start_time_in_seconds

                    runtimes_in_seconds[trial_num] = runtime_in_seconds

                # would like to see runtimes all in one matrix per eligibility criteria
                average_runtime_in_seconds = np.mean(runtimes_in_seconds)

                runtime_matrix[monthly_std_dev_index, monthly_mean_index] = average_runtime_in_seconds

                print( '\n\n[monthly mean, monthly standard deviation, min_req-base_sz_count]: [' + \
                        str(np.round(monthly_mean, 3)) + ', ' + str(np.round(monthly_std_dev, 3)) + ', ' + str(np.round(min_req_base_sz_count, 3)) + \
                        ']\naverage runtime: ' + str(np.round(average_runtime_in_seconds, 3)) )
                
                '''
                std_dev_runtime_in_seconds_str = str(np.round(np.std(runtimes_in_seconds), 3))
                runtime_statistical_summary_in_seconds = average_runtime_in_seconds_str + ' ± ' + std_dev_runtime_in_seconds_str + ' seconds'
                statistical_summary = '\n\n[monthly_mean, monthly_std_dev, min_req_base_sz_count]: [' \
                                        + str(monthly_mean) + ', ' + str(monthly_std_dev) + ', ' + str(min_req_base_sz_count) \
                                        + ']\n' + runtime_statistical_summary_in_seconds
    
                with open(file_path, 'a+') as text_file:

                    text_file.write(statistical_summary)

                print('\n' + runtime_statistical_summary_in_seconds + '\n\n')
                '''
        
        file_path = os.getcwd() + '/min_req_base_sz_count-' + str(min_req_base_sz_count) + '.txt'
        with open(file_path, 'a+') as text_file:

            text_file.write(pd.DataFrame(np.flipud(np.round(runtime_matrix, 3)), index = np.flip(monthly_std_dev_axis, 0), columns = monthly_mean_axis).to_string())

    #-------------------------------------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------------#
