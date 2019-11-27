import numpy as np
import scipy.stats as stats
from lifelines.statistics import logrank_test


def calculate_percent_changes(baseline_seizure_diaries,
                              testing_seizure_diaries,
                              num_patients_per_trial_arm):

    baseline_seizure_frequencies = np.mean(baseline_seizure_diaries, 1)
    testing_seizure_frequencies  = np.mean(testing_seizure_diaries, 1)

    '''
    for patient_index in range(num_patients_per_trial_arm):
        if(baseline_seizure_frequencies[patient_index] == 0):
            baseline_seizure_frequencies[patient_index] = 0.000001

    percent_changes = np.divide(baseline_seizure_frequencies - testing_seizure_frequencies, baseline_seizure_frequencies)
    '''

    baseline_seizure_frequencies[baseline_seizure_frequencies == 0] = 0.000001
    percent_changes = (baseline_seizure_frequencies - testing_seizure_frequencies)/baseline_seizure_frequencies

    return percent_changes


def calculate_time_to_prerandomizations(monthly_baseline_seizure_diaries,
                                        daily_testing_seizure_diaries,
                                        num_patients_per_trial_arm,
                                        num_testing_days):

    TTP_times      = np.zeros(num_patients_per_trial_arm)
    observed_array = np.zeros(num_patients_per_trial_arm)

    baseline_monthly_seizure_frequencies = np.mean(monthly_baseline_seizure_diaries, 1)

    for patient_index in range(num_patients_per_trial_arm):

        baseline_monthly_seizure_frequency = baseline_monthly_seizure_frequencies[patient_index]

        daily_testing_seizure_diary = daily_testing_seizure_diaries[patient_index]

        reached_count = False
        day_index = 0
        sum_count = 0

        while(not reached_count):

            sum_count = sum_count + daily_testing_seizure_diary[day_index]

            reached_count = sum_count >= baseline_monthly_seizure_frequency
            right_censored = day_index == (num_testing_days - 1)
            reached_count = reached_count or right_censored

            day_index = day_index + 1
        
        TTP_times[patient_index] = day_index
        observed_array[patient_index] = not right_censored

    return [TTP_times, observed_array]


def calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                   drug_arm_percent_changes):

    num_placebo_arm_responders     = np.sum(placebo_arm_percent_changes >= 0.5)
    num_drug_arm_responders        = np.sum(drug_arm_percent_changes    >= 0.5)
    num_placebo_arm_non_responders = len(placebo_arm_percent_changes) - num_placebo_arm_responders
    num_drug_arm_non_responders    = len(drug_arm_percent_changes)    - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])

    [_, RR50_p_value] = stats.fisher_exact(table)

    return RR50_p_value


def calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                     drug_arm_percent_changes):

    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)

    return MPC_p_value


def calculate_logrank_p_value(placebo_arm_TTP_times, 
                              placebo_arm_observed_array, 
                              drug_arm_TTP_times, 
                              drug_arm_observed_array):

    TTP_results = \
        logrank_test(placebo_arm_TTP_times,
                     drug_arm_TTP_times,
                     placebo_arm_observed_array,
                     drug_arm_observed_array)

    TTP_p_value = TTP_results.p_value

    return TTP_p_value


def calculate_trial_endpoints(num_testing_months,
                              num_theo_patients_per_trial_arm,
                              placebo_arm_baseline_monthly_seizure_diaries,
                              placebo_arm_testing_monthly_seizure_diaries,
                              placebo_arm_testing_daily_seizure_diaries,
                              drug_arm_baseline_monthly_seizure_diaries,
                              drug_arm_testing_monthly_seizure_diaries,
                              drug_arm_testing_daily_seizure_diaries):

    num_testing_days = num_testing_months*28

    placebo_arm_percent_changes = \
            calculate_percent_changes(placebo_arm_baseline_monthly_seizure_diaries,
                                      placebo_arm_testing_monthly_seizure_diaries,
                                      num_theo_patients_per_trial_arm)
    
    drug_arm_percent_changes = \
            calculate_percent_changes(drug_arm_baseline_monthly_seizure_diaries,
                                      drug_arm_testing_monthly_seizure_diaries,
                                      num_theo_patients_per_trial_arm)

    [placebo_arm_TTP_times, placebo_arm_observed_array] = \
            calculate_time_to_prerandomizations(placebo_arm_baseline_monthly_seizure_diaries,
                                                placebo_arm_testing_daily_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
    [drug_arm_TTP_times, drug_arm_observed_array] = \
            calculate_time_to_prerandomizations(drug_arm_baseline_monthly_seizure_diaries,
                                                drug_arm_testing_daily_seizure_diaries,
                                                num_theo_patients_per_trial_arm,
                                                num_testing_days)
    
    return [placebo_arm_percent_changes,
            drug_arm_percent_changes,
            placebo_arm_TTP_times, 
            placebo_arm_observed_array,
            drug_arm_TTP_times, 
            drug_arm_observed_array]
