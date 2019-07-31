import numpy as np
import scipy.stats as stats
import json
import os
import subprocess

def get_RR50_response_maps():

    expected_RR50_placebo_response_map_filename = 'expected_placebo_RR50_map_4'
    expected_RR50_drug_response_map_filename    = 'expected_drug_RR50_map_4'

    with open(os.getcwd() + '/' + expected_RR50_placebo_response_map_filename + '.json', 'r') as json_file:
        expected_RR50_placebo_response_map = np.array(json.load(json_file))/100

    with open(os.getcwd() + '/' + expected_RR50_drug_response_map_filename + '.json', 'r') as json_file:
        expected_RR50_drug_response_map = np.array(json.load(json_file))/100

    return [expected_RR50_placebo_response_map, expected_RR50_drug_response_map]


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


def calculate_expected_RR50_responses_from_maps(num_theo_patients_per_trial_arm,
                                      placebo_arm_patient_pop_monthly_param_sets,
                                      drug_arm_patient_pop_monthly_param_sets,
                                      expected_RR50_placebo_response_map,
                                      expected_RR50_drug_response_map):

    expected_RR50_placebo_response_per_monthly_param_set = np.zeros(num_theo_patients_per_trial_arm)
    expected_RR50_drug_response_per_monthly_param_set    = np.zeros(num_theo_patients_per_trial_arm)

    for patient_index in range(num_theo_patients_per_trial_arm):
        
        placebo_arm_monthly_mean    = int(placebo_arm_patient_pop_monthly_param_sets[patient_index, 0])
        placebo_arm_monthly_std_dev = int(placebo_arm_patient_pop_monthly_param_sets[patient_index, 1])
        drug_arm_monthly_mean       = int(drug_arm_patient_pop_monthly_param_sets[patient_index, 0])
        drug_arm_monthly_std_dev    = int(drug_arm_patient_pop_monthly_param_sets[patient_index, 1])

        expected_RR50_placebo_response_per_monthly_param_set[patient_index] = \
            expected_RR50_placebo_response_map[16 - placebo_arm_monthly_std_dev, placebo_arm_monthly_mean]
        expected_RR50_drug_response_per_monthly_param_set[patient_index] = \
            expected_RR50_drug_response_map[16 - drug_arm_monthly_std_dev, drug_arm_monthly_mean]
    
    expected_RR50_placebo_response = np.mean(expected_RR50_placebo_response_per_monthly_param_set)
    expected_RR50_drug_response    = np.mean(expected_RR50_drug_response_per_monthly_param_set)

    return [expected_RR50_placebo_response, expected_RR50_drug_response]


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
    
    num_placebo_responders     = np.sum(placebo_arm_percent_changes > 0.5)
    num_placebo_non_responders = num_patients_per_trial_arm - num_placebo_responders
    num_drug_responders        = np.sum(drug_arm_percent_changes > 0.5)
    num_drug_non_responders    = num_patients_per_trial_arm - num_drug_responders
    table = np.array([[num_placebo_responders, num_placebo_non_responders], [num_drug_responders,num_drug_non_responders]])
    [_, p_value] = stats.fisher_exact(table)

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
                                          num_trials,
                                          num_iter):

    p_value_array = np.zeros(num_trials)

    for trial_index in range(num_trials):

        print('trial #' + str(trial_index + 1) + ', iter #' + str(num_iter + 1))

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
    
    emp_stat_power = 100*np.sum(p_value_array < 0.05)/num_trials

    return emp_stat_power


def calculate_analytical_statistical_power(num_theo_patients_per_trial_arm,
                                           placebo_arm_patient_pop_monthly_param_sets,
                                           drug_arm_patient_pop_monthly_param_sets):

    [expected_RR50_placebo_response_map, expected_RR50_drug_response_map] = get_RR50_response_maps()

    [expected_RR50_placebo_response, expected_RR50_drug_response] = \
        calculate_expected_RR50_responses_from_maps(num_theo_patients_per_trial_arm,
                                                    placebo_arm_patient_pop_monthly_param_sets,
                                                    drug_arm_patient_pop_monthly_param_sets,
                                                    expected_RR50_placebo_response_map,
                                                    expected_RR50_drug_response_map)

    command = ['Rscript', 'Fisher_Exact_Power_Calc.R', str(expected_RR50_placebo_response), str(expected_RR50_drug_response), str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    fisher_exact_stat_power = 100*float(process.communicate()[0].decode().split()[1])

    return fisher_exact_stat_power


if(__name__=='__main__'):

    monthly_mean_min     = 4
    monthly_mean_max     = 16
    monthly_std_dev_min  = 1
    monthly_std_dev_max  = 8

    min_req_base_sz_count           = 4
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    num_theo_patients_per_trial_arm = 153

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials     = 500
    num_iterations = 100

    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient = num_baseline_days_per_patient + num_testing_days_per_patient

    fisher_exact_stat_power_array = np.zeros(num_iterations)
    emp_stat_power_array          = np.zeros(num_iterations)

    for num_iter in range(num_iterations):

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

        fisher_exact_stat_power = \
            calculate_analytical_statistical_power(num_theo_patients_per_trial_arm,
                                                   placebo_arm_patient_pop_monthly_param_sets,
                                                   drug_arm_patient_pop_monthly_param_sets)

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
                                                  num_trials,
                                                  num_iter)

        fisher_exact_stat_power_array[num_iter] = fisher_exact_stat_power
        emp_stat_power_array[num_iter]          = emp_stat_power
    
    
