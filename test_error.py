import numpy as np

def estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin,
                                num_months_per_patient_baseline, num_months_per_patient_testing,
                                min_required_baseline_seizures):
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
        
        5) num_months_per_patient_baseline:

            (int) - 
        
        6) num_months_per_patient_testing:

            (int) - 
        
        7) min_required_baseline_seizures:

            (int) - 

    Outputs:

        1) expected_RR50:

            (float) - 
        
        2) expected_MPC:

            (float) - 

    '''

    # calculate the total number of months
    num_months_per_patient_total = num_months_per_patient_baseline + num_months_per_patient_testing

    # check to see if the mean is not zero
    monthly_mu_not_zero = monthly_mu != 0

    # if the mean is not zero...
    if( monthly_mu_not_zero ):

        # check to see if the daily standard deviation is greater than the square root of the daily mean
        monthly_sigma_greater_than_square_root_of_monthly_mu = monthly_sigma > np.sqrt(monthly_mu)

        # if the daily standard deviation is greater than the square root of the daily mean...
        if( monthly_sigma_greater_than_square_root_of_monthly_mu ):

            # calculate the overdispersion parameter
            monthly_mu_squared = np.power(monthly_mu, 2)
            monthly_var = np.power(monthly_sigma, 2)
            monthly_alpha = (monthly_var - monthly_mu)/monthly_mu_squared

            # initialize the arrays that will store the RR50 and MPC from each trial
            RR50_array = np.zeros(num_trials_per_bin)
            MPC_array = np.zeros(num_trials_per_bin)

            # for every trial...
            for trial_index in range(num_trials_per_bin):

                # initialize the 2D array of monthly counts from multiple patients over one trial
                monthly_counts = np.zeros((num_patients_per_trial, num_months_per_patient_total))

                # for each individual patient
                for patient_index in range(num_patients_per_trial):

                    acceptable_baseline_rate = False
            
                    while(not acceptable_baseline_rate):

                        # for each month in each individual patient's diary
                        for month_index in range(num_months_per_patient_total):

                            # generate a monthly count according to gamma-poisson mixture
                            monthly_rate = np.random.gamma(1/monthly_alpha, monthly_alpha*monthly_mu)
                            monthly_count = np.random.poisson(monthly_rate)

                            # store the monthly count
                            monthly_counts[patient_index, month_index] = monthly_count 

                        if( np.sum(monthly_counts[patient_index, 0:num_months_per_patient_baseline]) >= min_required_baseline_seizures ):

                            acceptable_baseline_rate = True

                # separate the monthly counts into baseline and testing periods
                baseline_monthly_counts = monthly_counts[:, 0:num_months_per_patient_baseline]
                testing_monthly_counts = monthly_counts[:, num_months_per_patient_baseline:]
        
                # calculate the seizure frequencies of the baseline and test period for each patient
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


monthly_mu = 2.7
monthly_sigma = 1.65
num_patients_per_trial = 153
num_trials_per_bin = 500
num_months_per_patient_baseline = 2
num_months_per_patient_testing = 3
start_count = 0
stop_count = 7
decimal_round = 1

count_array = np.arange(start_count, stop_count + 1, 1)

for count_index in range(len(count_array)):

    min_required_baseline_seizures = count_array[count_index]

    [expected_RR50, expected_MPC] = \
        estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin,
                                    num_months_per_patient_baseline, num_months_per_patient_testing,
                                    min_required_baseline_seizures)

    RR50_string = 'expected RR50: ' + str(np.round(expected_RR50, decimal_round))
    MPC_string = 'expected MPC: ' + str(np.round(expected_MPC, decimal_round))

    data_string = '\n\nminimum required baseline seizures: ' + str(min_required_baseline_seizures) + '\n' + RR50_string + '\n' + MPC_string + '\n\n'

    print(data_string)

