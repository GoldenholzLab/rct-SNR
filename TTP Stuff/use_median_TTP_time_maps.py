import numpy as np
import json
import os


def retrieve_map(map_file_name, folder):

    map_file_path = folder + '/' + map_file_name + '.map'
    with open(map_file_path, 'r') as json_file:
        median_TTP_time_map = np.array(json.load(json_file))
    
    return median_TTP_time_map


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


def generate_patient_pop_hist(median_TTP_time_map_file_name, 
                              folder,
                              monthly_mean_min,
                              monthly_mean_max, 
                              monthly_std_dev_min, 
                              monthly_std_dev_max, 
                              num_theo_patients_per_trial_arm):

    median_TTP_time_map = \
        retrieve_map(median_TTP_time_map_file_name, folder)

    patient_pop_monthly_param_sets = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)


if(__name__=='__main__'):

    placebo_arm_median_TTP_time_map_file_name = 'placebo_arm_median_TTP_time_map'
    drug_arm_median_TTP_time_map_file_name    =    'drug_arm_median_TTP_time_map'
    folder = os.getcwd()

    monthly_mean_min     = 1
    monthly_mean_max     = 16
    monthly_std_dev_min  = 1
    monthly_std_dev_max  = 16

    num_theo_patients_per_trial_arm = 153

    placebo_arm_median_TTP_time_map = \
        retrieve_map(placebo_arm_median_TTP_time_map_file_name, folder)

    placebo_arm_patient_pop_monthly_param_sets = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)
    
    drug_arm_median_TTP_time_map = \
        retrieve_map(drug_arm_median_TTP_time_map_file_name, folder)
    
    drug_arm_patient_pop_monthly_param_sets = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)


    
    
