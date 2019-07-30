import numpy as np
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


def calculate_expected_RR50_responses(num_theo_patients_per_trial_arm,
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


def generate_pop_params(monthly_mean_min,    monthly_mean_max, 
                        monthly_std_dev_min, monthly_std_dev_max, 
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


if(__name__=='__main__'):

    monthly_mean_min     = 4
    monthly_mean_max     = 16
    monthly_std_dev_min  = 1
    monthly_std_dev_max  = 8

    num_theo_patients_per_trial_arm = 153

    [expected_RR50_placebo_response_map, expected_RR50_drug_response_map] = get_RR50_response_maps()

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

    [expected_RR50_placebo_response, expected_RR50_drug_response] = \
        calculate_expected_RR50_responses(num_theo_patients_per_trial_arm,
                                          placebo_arm_patient_pop_monthly_param_sets,
                                          drug_arm_patient_pop_monthly_param_sets,
                                          expected_RR50_placebo_response_map,
                                          expected_RR50_drug_response_map)

    command = ['Rscript', 'Fisher_Exact_Power_Calc.R', str(expected_RR50_placebo_response), str(expected_RR50_drug_response), str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    fisher_exact_stat_power = 100*float(process.communicate()[0].decode().split()[1])

    print(fisher_exact_stat_power)
