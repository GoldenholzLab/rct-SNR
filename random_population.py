import numpy as np


def generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                        monthly_std_dev_min, monthly_std_dev_max, 
                        num_patients_per_trial_arm):

    patient_pop_params = np.zeros((num_patients_per_trial_arm, 4))

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

        patient_pop_params[patient_index, 0] = monthly_mean
        patient_pop_params[patient_index, 1] = monthly_std_dev
        patient_pop_params[patient_index, 2] = daily_n
        patient_pop_params[patient_index, 3] = daily_odds_ratio
    
    return patient_pop_params


def generate_patient_diaries(patient_pop_params, num_patients_per_trial_arm, min_req_base_sz_count,
                             num_days_per_patient_baseline, num_days_per_patient_total):

    patient_diaries = np.zeros((num_patients_per_trial_arm, num_days_per_patient_total))

    for patient_index in range(num_patients_per_trial_arm):

        daily_n          = patient_pop_params[patient_index, 2]
        daily_odds_ratio = patient_pop_params[patient_index, 3]

        acceptable_baseline = False
        current_patient_diary = np.zeros(num_days_per_patient_total)

        while(not acceptable_baseline):

            for day_index in range(num_days_per_patient_total):

                daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
                daily_count = np.random.poisson(daily_rate)

                current_patient_diary[day_index] = daily_count
                
            current_patient_baseline_sz_count = np.sum(current_patient_diary[:num_days_per_patient_baseline])
                
            if(current_patient_baseline_sz_count >= min_req_base_sz_count):

                    acceptable_baseline = True
            
        patient_diaries[patient_index, :] = current_patient_diary
    
    return patient_diaries


def calculate_percent_changes(patient_diaries, num_days_per_patient_baseline, num_patients_per_trial_arm):
    
    baseline_daily_seizure_diaries = patient_diaries[:, :num_days_per_patient_baseline]
    testing_daily_seizure_diaries  = patient_diaries[:, num_days_per_patient_baseline:]
    
    baseline_daily_seizure_frequencies = np.mean(baseline_daily_seizure_diaries, 1)
    testing_daily_seizure_frequencies = np.mean(testing_daily_seizure_diaries, 1)

    for patient_index in range(num_patients_per_trial_arm):
        if(baseline_daily_seizure_frequencies[patient_index] == 0):
            baseline_daily_seizure_frequencies[patient_index] = 0.000000001
    
    percent_changes = np.divide(baseline_daily_seizure_frequencies - testing_daily_seizure_frequencies, baseline_daily_seizure_frequencies)

    return percent_changes


def calculate_times_to_prerandomization(patient_diaries):

    


if(__name__ == '__main__'):

    monthly_mean_min = 1
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16
    min_req_base_sz_count = 4

    num_patients_per_trial_arm = 153
    num_months_per_patient_baseline = 2
    num_months_per_patient_testing = 3
    num_trials = 20

    num_days_per_patient_baseline = num_months_per_patient_baseline*28
    num_days_per_patient_testing = num_months_per_patient_testing*28
    num_days_per_patient_total = num_days_per_patient_baseline + num_days_per_patient_testing

    patient_pop_params = \
        generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                            monthly_std_dev_min, monthly_std_dev_max, 
                            num_patients_per_trial_arm)

    for trial_index in range(num_trials):

        patient_diaries = \
            generate_patient_diaries(patient_pop_params, num_patients_per_trial_arm, min_req_base_sz_count,
                                     num_days_per_patient_baseline, num_days_per_patient_total)
        
        percent_changes = \
            calculate_percent_changes(patient_diaries, num_days_per_patient_baseline, num_patients_per_trial_arm)

        
        
            
        
