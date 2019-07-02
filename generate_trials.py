import numpy as np


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
                                  rct_constants_monthly_scale, effect_constants):
    '''

    Purpose:

        This function generates the patient diaries needed for one trial which is assumed to have a

        drug arm and a placebo arm. The patient populations of both the drug arm and the placebo arm

        both have the same number of patients.

    Inputs:

        1) monthly_mean:

            (float) - the mean of the monthly seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the monthly seizure counts in each patient's seizure diary
        
        3) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts
        
        4) rct_constants_monthly_scale:

            (1D Numpy array) - a numpy array containing the RCT design parameters for the trial to be generated

                               on a daily time scale. These RCT design parameters should be the same for all trials
                               
                               in each point on each map. This Numpy array contains the folllowing quantities:

                                    4.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    4.b) num_baseline_months:

                                        (int) - the number of baseline days in each patient's seizure diary

                                    4.c) num_testing_months:

                                        (int) - the number of testing days in each patient's seizure diary

        5) effect_constants:

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

    # convert the monthly parameters into daily parameters
    daily_mean = monthly_mean/28
    daily_std_dev = monthly_std_dev/np.sqrt(monthly_std_dev)

    # extract the RCT design parameters
    num_patients_per_trial_arm = rct_constants_monthly_scale[0]
    num_baseline_days          = rct_constants_monthly_scale[1]
    num_testing_days           = rct_constants_monthly_scale[2]

    # extract the statistical parameters for the drug and placebo effect
    placebo_mu    = effect_constants[0]
    placebo_sigma = effect_constants[1]
    drug_mu       = effect_constants[2]
    drug_sigma    = effect_constants[3]
    
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


<<<<<<< HEAD
def generate_one_iteration(monthly_mean, monthly_std_dev, min_req_base_sz_count, rct_constants_monthly_scale, effect_constants):
=======
def generate_one_iteration(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                           rct_constants_monthly_scale, effect_constants):
>>>>>>> parent of c857561... Update generate_trials.py
    
    '''

    Purpose:

    Inputs:
    
        1) monthly_mean:

            (float) - the mean of the monthly seizure counts in each patient's seizure diary

        2) monthly_std_dev:

            (float) - the standard deviation of the monthly seizure counts in each patient's seizure diary
        
        3) min_req_base_sz_count:
        
            (int) - the minimum number of required baseline seizure counts
        
        4) rct_constants_monthly_scale:

            (1D Numpy array) - a numpy array containing the RCT design parameters for the trial to be generated

                               on a daily time scale. These RCT design parameters should be the same for all trials
                               
                               in each point on each map. This Numpy array contains the folllowing quantities:

                                    4.a) num_patients_per_trial_arm:

                                        (int) - the number of patients generated per trial arm

                                    4.b) num_baseline_months:

                                        (int) - the number of baseline days in each patient's seizure diary

                                    4.c) num_testing_months:

                                        (int) - the number of testing days in each patient's seizure diary

        5) effect_constants:

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

        2) drug_arm_daily_seizure_diaries
        
            (2D Numpy array) - 

        3) first_type_1_arm_daily_seizure_diaries
        
            (2D Numpy array) - 

        4) second_type_1_arm_daily_seizure_diaries

            (2D Numpy array) - 

    '''

    effect_constants_placebo_vs_drug = effect_constants.copy()
    effect_constants_placebo_vs_placebo = np.zeros()

    placebo_mu    = effect_constants[0]
    placebo_sigma = effect_constants[1]
    effect_constants_placebo_vs_placebo[0] = placebo_mu
    effect_constants_placebo_vs_placebo[1] = placebo_sigma

    # generate one set of daily seizure diaries for each placebo arm and drug arm of a trial (two sets in total)
    [placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries] = \
        generate_one_trial_population(monthly_mean, monthly_std_dev, min_req_base_sz_count, 
                                      rct_constants_monthly_scale, effect_constants_placebo_vs_drug)
                
    # generate two sets of daily seizure diaries, both from a placebo arm (meant for calculating type 1 Error)
    [first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries] = \
        generate_one_trial_population(monthly_mean, monthly_std_dev, min_req_base_sz_count,
                                      rct_constants_monthly_scale, effect_constants_placebo_vs_placebo)

    
    
    return [placebo_arm_daily_seizure_diaries,      drug_arm_daily_seizure_diaries,
            first_type_1_arm_daily_seizure_diaries, second_type_1_arm_daily_seizure_diaries]


'''
def store_one_trial_population(directory, monthly_mean, monthly_std_dev, min_req_base_sz_count,
                               placebo_arm_daily_seizure_diaries, drug_arm_daily_seizure_diaries):

    folder = directory + '/eligibility_criteria__' + str(min_req_base_sz_count) + \
                         '/mean__'                 + str(monthly_mean) +          \
                         '/std_dev__'              + str(monthly_std_dev)
'''
