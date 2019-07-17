import numpy as np
import time


def generate_pop_params(monthly_mean_min,    
                        monthly_mean_max, 
                        monthly_std_dev_min, 
                        monthly_std_dev_max, 
                        num_points_per_trial_arm):

    patient_pop_monthly_params = np.zeros((num_points_per_trial_arm, 4))

    for patient_index in range(num_points_per_trial_arm):

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


def generate_daily_patient_diaries(daily_n,
                                   daily_odds_ratio,
                                   min_req_base_sz_count,
                                   num_days_per_patient_baseline, 
                                   num_days_per_patient_total,
                                   num_patients_per_point):

    daily_patient_diaries = np.zeros((num_patients_per_point, num_days_per_patient_total))

    for patient_index in range(num_patients_per_point):

        acceptable_baseline = False

        while(not acceptable_baseline):

            for day_index in range(num_days_per_patient_total):

                daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
                daily_count = np.random.poisson(daily_rate)

                daily_patient_diaries[patient_index, day_index] = daily_count
                
            patient_baseline_sz_count = np.sum(daily_patient_diaries[patient_index, :num_days_per_patient_baseline])
                
            if(patient_baseline_sz_count >= min_req_base_sz_count):

                acceptable_baseline = True
    
    return daily_patient_diaries
    

def count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies_per_point,
                                        testing_daily_seizure_diaries_per_point, 
                                        num_days_per_patient_testing,
                                        num_patients_per_point):

    TTP_times = np.zeros(num_patients_per_point)

    for patient_index in range(num_patients_per_point):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries_per_point[patient_index, day_index]

            reached_count = ( ( sum_count >= baseline_monthly_seizure_frequencies_per_point[patient_index] )  or ( day_index == (num_days_per_patient_testing - 1) ) )

            day_index = day_index + 1
    
        TTP_times[patient_index] = day_index

    return TTP_times


def calculate_individual_point_endpoints(daily_patient_diaries_per_point,
                                           num_patients_per_point, 
                                           num_days_per_patient_baseline, 
                                           num_days_per_patient_testing):
    
    baseline_daily_seizure_diaries_per_point = daily_patient_diaries_per_point[:, :num_days_per_patient_baseline]
    testing_daily_seizure_diaries_per_point  = daily_patient_diaries_per_point[:, num_days_per_patient_baseline:]
    
    baseline_daily_seizure_frequencies_per_point = np.mean(baseline_daily_seizure_diaries_per_point, 1)
    testing_daily_seizure_frequencies_per_point = np.mean(testing_daily_seizure_diaries_per_point, 1)

    baseline_monthly_seizure_frequencies_per_point = baseline_daily_seizure_frequencies_per_point*28

    for patient_index in range(num_patients_per_point):
        if(baseline_daily_seizure_frequencies_per_point[patient_index] == 0):
            baseline_daily_seizure_frequencies_per_point[patient_index] = 0.000000001
    
    percent_changes_per_point = np.divide(baseline_daily_seizure_frequencies_per_point - testing_daily_seizure_frequencies_per_point, baseline_daily_seizure_frequencies_per_point)

    TTP_times_per_point = count_days_to_prerandomization_time(baseline_monthly_seizure_frequencies_per_point,
                                                    testing_daily_seizure_diaries_per_point, 
                                                    num_days_per_patient_testing,
                                                    num_patients_per_point)

    return [percent_changes_per_point, TTP_times_per_point]


def apply_effect(daily_patient_diaries,
                 num_patients_per_point, 
                 num_days_per_patient_baseline, 
                 num_days_per_patient_testing, 
                 effect_mu,
                 effect_sigma):

    testing_daily_patient_diaries = daily_patient_diaries[:, num_days_per_patient_baseline:]

    for patient_index in range(num_patients_per_point):

        effect = np.random.normal(effect_mu, effect_sigma)
        if(effect > 1):
            effect = 1

        current_testing_daily_patient_diary = testing_daily_patient_diaries[patient_index, :]

        for day_index in range(num_days_per_patient_testing):

            current_seizure_count = current_testing_daily_patient_diary[day_index]
            num_removed = 0

            for seizure_index in range(np.int_(current_seizure_count)):

                if(np.random.random() <= np.abs(effect)):

                    num_removed = num_removed + np.sign(effect)
            
            current_seizure_count = current_seizure_count - num_removed
            current_testing_daily_patient_diary[day_index] = current_seizure_count

        testing_daily_patient_diaries[patient_index, :] = current_testing_daily_patient_diary
    
    daily_patient_diaries[:, num_days_per_patient_baseline:] = testing_daily_patient_diaries

    return daily_patient_diaries


def generate_point_endpoint_estimates(patient_pop_monthly_params,
                                      min_req_base_sz_count,
                                      num_days_per_patient_baseline, 
                                      num_days_per_patient_testing,
                                      num_days_per_patient_total,
                                      num_patients_per_point_per_trial_arm,
                                      num_points_per_trial_arm,
                                      placebo_mu,
                                      placebo_sigma,
                                      drug_mu,
                                      drug_sigma):

    median_percent_change_array = np.zeros(num_points_per_trial_arm)
    mean_TTP_array              = np.zeros(num_points_per_trial_arm)

    for point_index in range(num_points_per_trial_arm):

        patient_endpoint_estimation_start_time_in_seconds = time.time()

        daily_n          = patient_pop_monthly_params[point_index, 2]
        daily_odds_ratio = patient_pop_monthly_params[point_index, 3]

        daily_patient_diaries_per_point = \
            generate_daily_patient_diaries(daily_n,
                                           daily_odds_ratio,
                                           min_req_base_sz_count,
                                           num_days_per_patient_baseline, 
                                           num_days_per_patient_total,
                                           num_patients_per_point_per_trial_arm)

        # add the placebo effect
        daily_patient_diaries_per_point = \
            apply_effect(daily_patient_diaries_per_point,
                         num_patients_per_point_per_trial_arm, 
                         num_days_per_patient_baseline, 
                         num_days_per_patient_testing, 
                         placebo_mu,
                         placebo_sigma)
        
        # add the drug efficacy
        daily_patient_diaries_per_point = \
            apply_effect(daily_patient_diaries_per_point,
                         num_patients_per_point_per_trial_arm, 
                         num_days_per_patient_baseline, 
                         num_days_per_patient_testing, 
                         drug_mu,
                         drug_sigma)

        [percent_changes_per_point, TTP_times_per_point] = \
            calculate_individual_point_endpoints(daily_patient_diaries_per_point,
                                                   num_patients_per_point_per_trial_arm, 
                                                   num_days_per_patient_baseline, 
                                                   num_days_per_patient_testing)

        # median percent change per point was used instead of mean percent change because the histogram of percent change showed that it was left-skewed
        median_percent_change_per_point = np.median(percent_changes_per_point)
        mean_TTP_per_point = np.mean(TTP_times_per_point)

        median_percent_change_array[point_index] = median_percent_change_per_point
        mean_TTP_array[point_index] = mean_TTP_per_point

        patient_endpoint_estimation_stop_time_in_seconds = time.time()
        patient_endpoint_estimation_total_runtime_in_seconds = patient_endpoint_estimation_stop_time_in_seconds - patient_endpoint_estimation_start_time_in_seconds

        point_index_str                                          = 'point index: ' + str(point_index + 1)
        median_percent_change_per_point_str                      = 'median percent change per point:              ' + str(np.round(100*median_percent_change_per_point, 3)) + ' %'
        mean_TTP_per_point_str                                   = 'average time-to-prerandomization per point:   ' + str(np.round(mean_TTP_per_point, 3)) + ' days'
        patient_endpoint_estimation_total_runtime_in_minutes_str = 'individual point endpoint estimation runtime: ' +  str(np.round(patient_endpoint_estimation_total_runtime_in_seconds, 3)) + ' seconds'

        data_str = '\n' + point_index_str + '\n' + median_percent_change_per_point_str + '\n' + mean_TTP_per_point_str + '\n' + patient_endpoint_estimation_total_runtime_in_minutes_str + '\n'

        print(data_str)
    
    return [median_percent_change_array, mean_TTP_array]


if(__name__=='__main__'):

    monthly_mean_min           = 4
    monthly_mean_max           = 16
    monthly_std_dev_min        = 1
    monthly_std_dev_max        = 8

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    min_req_base_sz_count = 4
    num_months_per_patient_baseline = 2
    num_months_per_patient_testing = 3
    num_points_per_trial_arm = 153

    num_patients_per_point_per_trial_arm = 5000

    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#

    num_days_per_patient_baseline = 28*num_months_per_patient_baseline
    num_days_per_patient_testing = 28*num_months_per_patient_testing
    num_days_per_patient_total = num_days_per_patient_baseline + num_days_per_patient_testing

    algorithm_start_time_in_seconds = time.time()

    patient_placebo_pop_monthly_params = \
        generate_pop_params(monthly_mean_min,    
                            monthly_mean_max, 
                            monthly_std_dev_min, 
                            monthly_std_dev_max, 
                            num_points_per_trial_arm)
    
    patient_drug_pop_monthly_params = \
        generate_pop_params(monthly_mean_min,    
                            monthly_mean_max, 
                            monthly_std_dev_min, 
                            monthly_std_dev_max, 
                            num_points_per_trial_arm)

    [median_placebo_percent_change_array, mean_placebo_TTP_array] = \
        generate_point_endpoint_estimates(patient_placebo_pop_monthly_params,
                                          min_req_base_sz_count,
                                          num_days_per_patient_baseline, 
                                          num_days_per_patient_testing,
                                          num_days_per_patient_total,
                                          num_patients_per_point_per_trial_arm,
                                          num_points_per_trial_arm,
                                          placebo_mu,
                                          placebo_sigma,
                                          0,
                                          0)
    
    [median_drug_percent_change_array, mean_drug_TTP_array] = \
        generate_point_endpoint_estimates(patient_drug_pop_monthly_params,
                                          min_req_base_sz_count,
                                          num_days_per_patient_baseline, 
                                          num_days_per_patient_testing,
                                          num_days_per_patient_total,
                                          num_patients_per_point_per_trial_arm,
                                          num_points_per_trial_arm,
                                          placebo_mu,
                                          placebo_sigma,
                                          drug_mu,
                                          drug_sigma)


    algorithm_stop_time_in_seconds  = time.time()
    algorithm_total_runtime_in_minutes  = (algorithm_stop_time_in_seconds - algorithm_start_time_in_seconds)/60
    algorithm_total_runtime_in_minutes_str = 'endpoint estimation overall runtime: ' + str(np.round(algorithm_total_runtime_in_minutes, 3)) + ' minutes'

    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------#
