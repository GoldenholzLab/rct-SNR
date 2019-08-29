import numpy as np
import scipy.stats as stats
import time


def generate_patient_pop_params(monthly_mean_min,
                                monthly_mean_max, 
                                monthly_std_dev_min, 
                                monthly_std_dev_max, 
                                num_theo_patients_per_trial_arm):

    patient_pop_monthly_param_sets = np.zeros((num_theo_patients_per_trial_arm , 2))

    for theo_patient_index in range(num_theo_patients_per_trial_arm ):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        patient_pop_monthly_param_sets[theo_patient_index, 0] = monthly_mean
        patient_pop_monthly_param_sets[theo_patient_index, 1] = monthly_std_dev

    return patient_pop_monthly_param_sets


def generate_one_trial_arm_of_patient_diaries(patient_pop_monthly_param_sets, 
                                              num_patients_per_trial_arm, 
                                              min_req_base_sz_count,
                                              num_baseline_days_per_patient, 
                                              num_total_days_per_patient):

    daily_patient_diaries = np.zeros((num_patients_per_trial_arm, num_total_days_per_patient))

    for patient_index in range(num_patients_per_trial_arm):

        monthly_mean    = patient_pop_monthly_param_sets[patient_index, 0]
        monthly_std_dev = patient_pop_monthly_param_sets[patient_index, 1]

        daily_mean = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)
        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)

        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        acceptable_baseline = False
        current_daily_patient_diary = np.zeros(num_total_days_per_patient)

        while(not acceptable_baseline):

            for day_index in range(num_total_days_per_patient):

                daily_rate = np.random.gamma(daily_n, daily_odds_ratio)
                daily_count = np.random.poisson(daily_rate)

                current_daily_patient_diary[day_index] = daily_count
                
            current_patient_baseline_sz_count = np.sum(current_daily_patient_diary[:num_baseline_days_per_patient])
                
            if(current_patient_baseline_sz_count >= min_req_base_sz_count):

                    acceptable_baseline = True
            
        daily_patient_diaries[patient_index, :] = current_daily_patient_diary
    
    return daily_patient_diaries


def calculate_individual_patient_percent_changes_per_diary_set(daily_patient_diaries,
                                                               num_patients_per_trial_arm, 
                                                               num_baseline_days_per_patient):
    
    baseline_daily_seizure_diaries = daily_patient_diaries[:, :num_baseline_days_per_patient]
    testing_daily_seizure_diaries  = daily_patient_diaries[:, num_baseline_days_per_patient:]
    
    baseline_daily_seizure_frequencies = np.mean(baseline_daily_seizure_diaries, 1)
    testing_daily_seizure_frequencies = np.mean(testing_daily_seizure_diaries, 1)

    for patient_index in range(num_patients_per_trial_arm):
        if(baseline_daily_seizure_frequencies[patient_index] == 0):
            baseline_daily_seizure_frequencies[patient_index] = 0.000000001
    
    percent_changes = np.divide(baseline_daily_seizure_frequencies - testing_daily_seizure_frequencies, baseline_daily_seizure_frequencies)

    return percent_changes


def apply_effect(daily_patient_diaries,
                 num_patients_per_trial_arm, 
                 num_baseline_days_per_patient,
                 num_testing_days_per_patient, 
                 effect_mu,
                 effect_sigma):

    testing_daily_patient_diaries = daily_patient_diaries[:,  num_baseline_days_per_patient:]

    for patient_index in range(num_patients_per_trial_arm):

        effect = np.random.normal(effect_mu, effect_sigma)
        if(effect > 1):
            effect = 1

        current_testing_daily_patient_diary = testing_daily_patient_diaries[patient_index, :]

        for day_index in range(num_testing_days_per_patient):

            current_seizure_count = current_testing_daily_patient_diary[day_index]
            num_removed = 0

            for seizure_index in range(np.int_(current_seizure_count)):

                if(np.random.random() <= np.abs(effect)):

                    num_removed = num_removed + np.sign(effect)
            
            current_seizure_count = current_seizure_count - num_removed
            current_testing_daily_patient_diary[day_index] = current_seizure_count

        testing_daily_patient_diaries[patient_index, :] = current_testing_daily_patient_diary
    
    daily_patient_diaries[:, num_baseline_days_per_patient:] = testing_daily_patient_diaries

    return daily_patient_diaries


def generate_individual_patient_percent_changes_per_trial_arm(patient_pop_monthly_param_sets, 
                                                              num_patients_per_trial_arm,
                                                              num_baseline_days_per_patient,
                                                              num_testing_days_per_patient,
                                                              num_total_days_per_patient,
                                                              min_req_base_sz_count,
                                                              placebo_mu,
                                                              placebo_sigma,
                                                              drug_mu,
                                                              drug_sigma):

    one_trial_arm_daily_seizure_diaries = \
        generate_one_trial_arm_of_patient_diaries(patient_pop_monthly_param_sets, 
                                                  num_patients_per_trial_arm, 
                                                  min_req_base_sz_count,
                                                  num_baseline_days_per_patient, 
                                                  num_total_days_per_patient)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_patients_per_trial_arm, 
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     placebo_mu,
                     placebo_sigma)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_patients_per_trial_arm, 
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     drug_mu,
                     drug_sigma)

    percent_changes = \
        calculate_individual_patient_percent_changes_per_diary_set(one_trial_arm_daily_seizure_diaries,
                                                                   num_patients_per_trial_arm, 
                                                                   num_baseline_days_per_patient)
    
    return percent_changes


def calculate_one_trial_p_value(placebo_arm_patient_pop_monthly_param_sets,
                                drug_arm_patient_pop_monthly_param_sets,
                                num_patients_per_trial_arm,
                                num_baseline_days_per_patient,
                                num_testing_days_per_patient,
                                num_total_days_per_patient,
                                min_req_base_sz_count,
                                placebo_mu,
                                placebo_sigma,
                                drug_mu,
                                drug_sigma):

    placebo_arm_percent_changes = \
        generate_individual_patient_percent_changes_per_trial_arm(placebo_arm_patient_pop_monthly_param_sets, 
                                                                  num_patients_per_trial_arm,
                                                                  num_baseline_days_per_patient,
                                                                  num_testing_days_per_patient,
                                                                  num_total_days_per_patient,
                                                                  min_req_base_sz_count,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  0,
                                                                  0)
    
    drug_arm_percent_changes = \
        generate_individual_patient_percent_changes_per_trial_arm(placebo_arm_patient_pop_monthly_param_sets, 
                                                                  num_patients_per_trial_arm,
                                                                  num_baseline_days_per_patient,
                                                                  num_testing_days_per_patient,
                                                                  num_total_days_per_patient,
                                                                  min_req_base_sz_count,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  drug_mu,
                                                                  drug_sigma)
    
    [_, p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)

    return p_value


def calculate_empirical_statistical_power(placebo_arm_patient_pop_monthly_param_sets,
                                          drug_arm_patient_pop_monthly_param_sets,
                                          num_patients_per_trial_arm,
                                          num_baseline_days_per_patient,
                                          num_testing_days_per_patient,
                                          num_total_days_per_patient,
                                          min_req_base_sz_count,
                                          placebo_mu,
                                          placebo_sigma,
                                          drug_mu,
                                          drug_sigma,
                                          num_trials):

    p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        trial_start_time_in_seconds = time.time()

        p_value_array[trial_index] = \
            calculate_one_trial_p_value(placebo_arm_patient_pop_monthly_param_sets,
                                        drug_arm_patient_pop_monthly_param_sets,
                                        num_patients_per_trial_arm,
                                        num_baseline_days_per_patient,
                                        num_testing_days_per_patient,
                                        num_total_days_per_patient,
                                        min_req_base_sz_count,
                                        placebo_mu,
                                        placebo_sigma,
                                        drug_mu,
                                        drug_sigma)
        
        trial_stop_time_in_seconds = time.time()
        trial_runtime_str = str(np.round(trial_stop_time_in_seconds - trial_start_time_in_seconds, 3))
        print( 'trial #' + str(trial_index + 1) + ', runtime: ' + trial_runtime_str + ' seconds' )
    
    emp_stat_power = 100*np.sum(p_value_array < 0.05)/num_trials

    return emp_stat_power


if(__name__=='__main__'):

    monthly_mean_min                = 4
    monthly_mean_max                = 16
    monthly_std_dev_min             = 1
    monthly_std_dev_max             = 8
    num_theo_patients_per_trial_arm = 153
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    min_req_base_sz_count           = 4
    placebo_mu                      = 0
    placebo_sigma                   = 0.05
    drug_mu                         = 0.2
    drug_sigma                      = 0.05
    num_trials                      = 100

    placebo_arm_patient_pop_monthly_param_sets = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)
    
    drug_arm_patient_pop_monthly_param_sets = \
            generate_patient_pop_params(monthly_mean_min,
                                        monthly_mean_max, 
                                        monthly_std_dev_min, 
                                        monthly_std_dev_max, 
                                        num_theo_patients_per_trial_arm)

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    emp_stat_power = \
        calculate_empirical_statistical_power(placebo_arm_patient_pop_monthly_param_sets,
                                              drug_arm_patient_pop_monthly_param_sets,
                                              num_theo_patients_per_trial_arm,
                                              num_baseline_days_per_patient,
                                              num_testing_days_per_patient,
                                              num_total_days_per_patient,
                                              min_req_base_sz_count,
                                              placebo_mu,
                                              placebo_sigma,
                                              drug_mu,
                                              drug_sigma,
                                              num_trials)

    print(emp_stat_power)

