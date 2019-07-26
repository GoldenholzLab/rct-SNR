import numpy as np
import matplotlib.pyplot as plt

'''

This script revealed that a hazard ratio of apporixmately exp{0.285} is reasonable to expect for the

population being used, although it it goes down slightly to exp{0.105} for the wider population.

'''

def generate_patient_pop_params(monthly_mean_min,
                                monthly_mean_max, 
                                monthly_std_dev_min, 
                                monthly_std_dev_max, 
                                num_theo_patients_per_trial_arm ):

    patient_monthly_params = np.zeros((num_theo_patients_per_trial_arm , 2))
    patient_daily_params = np.zeros((num_theo_patients_per_trial_arm , 2))

    for theo_patient_index in range(num_theo_patients_per_trial_arm ):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        daily_mean    = monthly_mean/28
        daily_std_dev = monthly_std_dev/np.sqrt(28)

        daily_var = np.power(daily_std_dev, 2)
        daily_overdispersion = (daily_var - daily_mean)/np.power(daily_mean, 2)
        daily_n = 1/daily_overdispersion
        daily_odds_ratio = daily_overdispersion*daily_mean

        patient_monthly_params[theo_patient_index, 0] = monthly_mean
        patient_monthly_params[theo_patient_index, 1] = monthly_std_dev

        patient_daily_params[theo_patient_index, 0] = daily_n
        patient_daily_params[theo_patient_index, 1] = daily_odds_ratio

    return [patient_monthly_params, patient_daily_params]


def generate_daily_seizure_diary(daily_n, 
                                 daily_odds_ratio,
                                 num_baseline_days_per_patient,
                                 num_total_days_per_patient,
                                 min_req_bs_sz_count):
    
    seizure_diary = np.zeros(num_total_days_per_patient)
    acceptable_baseline = False

    while (not acceptable_baseline):

        for day_index in range(num_total_days_per_patient):

            daily_rate  = np.random.gamma(daily_n, daily_odds_ratio)
            daily_count = np.random.poisson(daily_rate)

            seizure_diary[day_index] = daily_count
    
        baseline_sz_count = np.sum(seizure_diary[:num_baseline_days_per_patient])

        if(baseline_sz_count >= min_req_bs_sz_count):

            acceptable_baseline = True
    
    return seizure_diary


def generate_one_trial_arm_of_seizure_diaries(patient_daily_params, 
                                              num_theo_patients_per_trial_arm,
                                              num_baseline_days_per_patient,
                                              num_total_days_per_patient,
                                              min_req_bs_sz_count):

    one_trial_arm_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_total_days_per_patient))

    for patient_index in range(num_theo_patients_per_trial_arm):

        daily_n          = patient_daily_params[patient_index, 0]
        daily_odds_ratio = patient_daily_params[patient_index, 1]

        one_trial_arm_seizure_diaries[patient_index, :] = \
            generate_daily_seizure_diary(daily_n, 
                                         daily_odds_ratio,
                                         num_baseline_days_per_patient,
                                         num_total_days_per_patient,
                                         min_req_bs_sz_count)
    
    return one_trial_arm_seizure_diaries


def calculate_individual_patient_TTP_times_per_diary_set(daily_seizure_diaries,
                                                         num_baseline_days_per_patient,
                                                         num_testing_days_per_patient,
                                                         num_daily_seizure_diaries):

    baseline_daily_seizure_diaries = daily_seizure_diaries[:, :num_baseline_days_per_patient]
    testing_daily_seizure_diaries  = daily_seizure_diaries[:, num_baseline_days_per_patient:]

    baseline_monthly_seizure_frequencies = 28*np.mean(baseline_daily_seizure_diaries, 1)

    TTP_times = np.zeros(num_daily_seizure_diaries)

    for daily_seizure_diary_index in range(num_daily_seizure_diaries):

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + testing_daily_seizure_diaries[daily_seizure_diary_index, day_index]
            reached_count = ( ( sum_count >= baseline_monthly_seizure_frequencies[daily_seizure_diary_index] )  or ( day_index == (num_testing_days_per_patient - 1) ) )
            day_index = day_index + 1
        
        TTP_times[daily_seizure_diary_index] = day_index
    
    return TTP_times


def apply_effect(daily_seizure_diaries,
                 num_baseline_days_per_patient,
                 num_testing_days_per_patient,
                 num_seizure_diaries,
                 effect_mu,
                 effect_sigma):
    
    testing_daily_seizure_diaries  = daily_seizure_diaries[:, num_baseline_days_per_patient:]

    for seizure_diary_index in range(num_seizure_diaries):

        effect = np.random.normal(effect_mu, effect_sigma)
        if(effect > 1):
            effect = 1
        
        current_testing_daily_seizure_diary = testing_daily_seizure_diaries[seizure_diary_index, :]

        for day_index in range(num_testing_days_per_patient):

            current_seizure_count = current_testing_daily_seizure_diary[day_index]
            num_removed = 0

            for seizure_index in range(np.int_(current_seizure_count)):

                if(np.random.random() <= np.abs(effect)):

                    num_removed = num_removed + np.sign(effect)
                
            current_seizure_count = current_seizure_count - num_removed
            current_testing_daily_seizure_diary[day_index] = current_seizure_count
        
        testing_daily_seizure_diaries[seizure_diary_index, :] = current_testing_daily_seizure_diary

    daily_seizure_diaries[:, num_baseline_days_per_patient:] = testing_daily_seizure_diaries

    return daily_seizure_diaries


def generate_individual_patient_TTP_times_per_trial_arm(patient_pop_daily_params, 
                                                        num_theo_patients_per_trial_arm,
                                                        num_baseline_days_per_patient,
                                                        num_testing_days_per_patient,
                                                        num_total_days_per_patient,
                                                        min_req_bs_sz_count,
                                                        placebo_mu,
                                                        placebo_sigma,
                                                        drug_mu,
                                                        drug_sigma):

    one_trial_arm_daily_seizure_diaries = \
        generate_one_trial_arm_of_seizure_diaries(patient_pop_daily_params, 
                                                  num_theo_patients_per_trial_arm,
                                                  num_baseline_days_per_patient,
                                                  num_total_days_per_patient,
                                                  min_req_bs_sz_count)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_theo_patients_per_trial_arm,
                     placebo_mu,
                     placebo_sigma)
        
    one_trial_arm_daily_seizure_diaries = \
        apply_effect(one_trial_arm_daily_seizure_diaries,
                     num_baseline_days_per_patient,
                     num_testing_days_per_patient,
                     num_theo_patients_per_trial_arm,
                     drug_mu,
                     drug_sigma)

    one_trial_arm_TTP_times = \
        calculate_individual_patient_TTP_times_per_diary_set(one_trial_arm_daily_seizure_diaries,
                                                             num_baseline_days_per_patient,
                                                             num_testing_days_per_patient,
                                                             num_theo_patients_per_trial_arm)

    return one_trial_arm_TTP_times


def generate_trial_TTP_times(monthly_mean_min,
                             monthly_mean_max, 
                             monthly_std_dev_min, 
                             monthly_std_dev_max,
                             min_req_bs_sz_count,
                             num_baseline_months_per_patient,
                             num_testing_months_per_patient,
                             num_theo_patients_per_trial_arm,
                             placebo_mu,
                             placebo_sigma,
                             drug_mu,
                             drug_sigma):

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    [_, patient_placebo_pop_daily_params] = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)

    [_, patient_drug_pop_daily_params] = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)

    one_placebo_arm_TTP_times = \
        generate_individual_patient_TTP_times_per_trial_arm(patient_placebo_pop_daily_params, 
                                                            num_theo_patients_per_trial_arm,
                                                            num_baseline_days_per_patient,
                                                            num_testing_days_per_patient,
                                                            num_total_days_per_patient,
                                                            min_req_bs_sz_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            0,
                                                            0)
    
    one_drug_arm_TTP_times = \
        generate_individual_patient_TTP_times_per_trial_arm(patient_drug_pop_daily_params, 
                                                            num_theo_patients_per_trial_arm,
                                                            num_baseline_days_per_patient,
                                                            num_testing_days_per_patient,
                                                            num_total_days_per_patient,
                                                            min_req_bs_sz_count,
                                                            placebo_mu,
                                                            placebo_sigma,
                                                            drug_mu,
                                                            drug_sigma)
    
    return [one_placebo_arm_TTP_times, one_drug_arm_TTP_times]


if(__name__=='__main__'):

    monthly_mean_min = 4
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    min_req_bs_sz_count             = 4
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    num_theo_patients_per_trial_arm = 10000
    
    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    [one_placebo_arm_TTP_times, one_drug_arm_TTP_times] = \
        generate_trial_TTP_times(monthly_mean_min,
                                 monthly_mean_max, 
                                 monthly_std_dev_min, 
                                 monthly_std_dev_max,
                                 min_req_bs_sz_count,
                                 num_baseline_months_per_patient,
                                 num_testing_months_per_patient,
                                 num_theo_patients_per_trial_arm,
                                 placebo_mu,
                                 placebo_sigma,
                                 drug_mu,
                                 drug_sigma)
    
    num_testing_days_per_patient   = 28*num_testing_months_per_patient
    one_placebo_arm_survival_data  = np.zeros((2, num_testing_days_per_patient - 1))
    one_drug_arm_survival_data     = np.zeros((2, num_testing_days_per_patient - 1))
    one_placebo_arm_survival_curve = np.zeros(num_testing_days_per_patient - 1)
    one_drug_arm_survival_curve    = np.zeros(num_testing_days_per_patient - 1)

    one_placebo_arm_survival_data[0, :]  = np.arange(1, num_testing_days_per_patient)
    one_drug_arm_survival_data[0, :]     = np.arange(1, num_testing_days_per_patient)

    for day in range(num_testing_days_per_patient - 1):
        one_placebo_arm_survival_data[1, day] = np.sum(one_placebo_arm_TTP_times == one_placebo_arm_survival_data[0, day])
        one_drug_arm_survival_data[1, day]    = np.sum(one_drug_arm_TTP_times    == one_drug_arm_survival_data[0, day])

    one_placebo_arm_survival_curve = num_theo_patients_per_trial_arm  - np.cumsum(one_placebo_arm_survival_data[1, :])
    one_drug_arm_survival_curve   = num_theo_patients_per_trial_arm  - np.cumsum(one_drug_arm_survival_data[1, :])

    hazard_ratios = np.divide(one_drug_arm_survival_curve, one_placebo_arm_survival_curve)
    log_hazard_ratios = np.log(hazard_ratios)
    average_log_hazard_ratio = np.mean(log_hazard_ratios)
    std_dev_log_hazard_ratio = np.std(log_hazard_ratios)

    plt.figure()
    plt.plot(np.arange(1, num_testing_days_per_patient), one_placebo_arm_survival_curve)
    plt.plot(np.arange(1, num_testing_days_per_patient), one_drug_arm_survival_curve)
    plt.show()

    print(log_hazard_ratios)
    print(str(np.round(average_log_hazard_ratio, 3)) + ' ± ' + str(np.round(std_dev_log_hazard_ratio, 3)))
