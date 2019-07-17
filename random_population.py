import numpy as np


def generate_pop_params(monthly_mean_min,    
                        monthly_mean_max, 
                        monthly_std_dev_min, 
                        monthly_std_dev_max, 
                        num_patients_per_trial_arm):

    patient_pop_monthly_params = np.zeros((num_patients_per_trial_arm, 4))

    for patient_index in range(num_patients_per_trial_arm):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        daily_mean = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)
        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)

        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        patient_pop_monthly_params[patient_index, 0] = monthly_mean
        patient_pop_monthly_params[patient_index, 1] = monthly_std_dev
        patient_pop_monthly_params[patient_index, 2] = daily_n
        patient_pop_monthly_params[patient_index, 3] = daily_odds_ratio
    
    return patient_pop_monthly_params


def generate_patient_diary(daily_n, daily_odds_ratio, 
                           min_req_base_sz_count,
                           num_days_per_patient_baseline, 
                           num_days_per_patient_total):

    acceptable_baseline = False
    daily_patient_diary = np.zeros(num_days_per_patient_total)

    while(not acceptable_baseline):

        for day_index in range(num_days_per_patient_total):

            daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
            daily_count = np.random.poisson(daily_rate)

            daily_patient_diary[day_index] = daily_count
                
        patient_baseline_sz_count = np.sum(daily_patient_diary[:num_days_per_patient_baseline])
                
        if(patient_baseline_sz_count >= min_req_base_sz_count):

            acceptable_baseline = True
    
    return daily_patient_diary
    

def count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies, 
                                        testing_daily_seizure_diaries, 
                                        num_days_per_patient_testing,
                                        num_patients_per_trial_arm):

    TTP_times = np.zeros(num_patients_per_trial_arm)

    for patient_index in range(num_patients_per_trial_arm):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries[patient_index, day_index]

            reached_count = ( ( sum_count >= baseline_monthly_seizure_frequencies[patient_index] )  or ( day_index == (num_days_per_patient_testing - 1) ) )

            day_index = day_index + 1
    
        TTP_times[patient_index] = day_index

    return TTP_times


if(__name__=='__main__'):

    

'''
def calculate_individual_point_measures(daily_patient_diaries_per_point, 
                                        num_patients_per_trial_arm,
                                        num_patients_per_point, 
                                        num_days_per_patient_baseline,
                                        num_days_per_patient_testing):

    baseline_daily_seizure_diaries_per_point = daily_patient_diaries_per_point[:, :num_days_per_patient_baseline]
    testing_daily_seizure_diaries_per_point  = daily_patient_diaries_per_point[:, num_days_per_patient_baseline:]

    baseline_daily_seizure_frequencies_per_point = np.mean(baseline_daily_seizure_diaries_per_point, 1)
    testing_daily_seizure_frequencies_per_point = np.mean(testing_daily_seizure_diaries_per_point, 1)

    baseline_monthly_seizure_frequencies_per_point = baseline_daily_seizure_frequencies_per_point*28

    for patient_point_iter in range(num_patients_per_point):
        if(baseline_daily_seizure_frequencies_per_point[patient_point_iter] == 0):
            baseline_daily_seizure_frequencies_per_point[patient_point_iter] = 0.000000001

    percent_changes_per_point_per_trial_arm = np.divide(baseline_daily_seizure_frequencies_per_point - testing_daily_seizure_frequencies_per_point, baseline_daily_seizure_frequencies_per_point)

    TTP_times_per_point_per_trial_arm = \
        count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies_per_point,
                                            testing_daily_seizure_diaries_per_point, 
                                            num_days_per_patient_testing,
                                            num_patients_per_trial_arm)

    median_percent_change_per_point_per_trial_arm = np.median(percent_changes_per_point_per_trial_arm)
    median_TTP_time_per_point_per_trial_arm       = np.median(TTP_times_per_point_per_trial_arm)

    return [median_percent_change_per_point_per_trial_arm, median_TTP_time_per_point_per_trial_arm]
'''
'''
if(__name__=='__main__'):

    monthly_mean_min = 4
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_months_per_patient_baseline = 2
    num_months_per_patient_testing = 3
    num_patients_per_trial_arm = 153
    min_req_base_sz_count = 4

    num_patients_per_point = 5000


    #---------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------------#
    
    num_days_per_patient_baseline = num_months_per_patient_baseline*28
    num_days_per_patient_testing = num_months_per_patient_testing*28
    num_days_per_patient_total = num_days_per_patient_baseline + num_days_per_patient_testing

    patient_pop_monthly_params = \
        generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                            monthly_std_dev_min, monthly_std_dev_max, 
                            num_patients_per_trial_arm)
    
    #median_percent_changes_per_trial_arm

    for patient_index in range(num_patients_per_trial_arm):

        daily_n          = patient_pop_monthly_params[patient_index, 2]
        daily_odds_ratio = patient_pop_monthly_params[patient_index, 3]

        daily_patient_diaries_per_point_per_trial_arm = np.zeros((num_patients_per_point, num_days_per_patient_total))

        for patient_point_inter in range(num_patients_per_point):

            daily_patient_diaries_per_point_per_trial_arm[patient_point_inter, :] = \
                generate_patient_diary(daily_n, daily_odds_ratio, min_req_base_sz_count,
                                       num_days_per_patient_baseline, num_days_per_patient_total)
    
        # median of percent change was done here instead of mean of percent change because histogram of percent change per point showed a left-skewed distribution
        # for TTP, mean and median were about the same
        [median_percent_change_per_point_per_trial_arm, median_TTP_time_per_point_per_trial_arm] = \
            calculate_individual_point_measures(daily_patient_diaries_per_point_per_trial_arm, 
                                                num_patients_per_trial_arm,
                                                num_patients_per_point, 
                                                num_days_per_patient_baseline,
                                                num_days_per_patient_testing)

    #---------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------------#
'''