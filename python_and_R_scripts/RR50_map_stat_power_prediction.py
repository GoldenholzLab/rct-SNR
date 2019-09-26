import numpy as np
import json
import subprocess


def get_RR50_response_maps(maps_folder):

    expected_RR50_placebo_response_map_filename = 'expected_RR50_placebo_arm_map.json'
    expected_RR50_drug_response_map_filename    = 'expected_RR50_drug_arm_map.json'

    with open(maps_folder + '/' + expected_RR50_placebo_response_map_filename, 'r') as json_file:
        expected_RR50_placebo_response_map = np.array(json.load(json_file))

    with open(maps_folder + '/' + expected_RR50_drug_response_map_filename, 'r') as json_file:
        expected_RR50_drug_response_map = np.array(json.load(json_file))

    return [expected_RR50_placebo_response_map, expected_RR50_drug_response_map]


def predict_RR50_placebo_and_drug_arm_responses(monthly_mean_min,
                                                monthly_mean_max,
                                                monthly_std_dev_min,
                                                monthly_std_dev_max,
                                                theo_trial_arm_patient_pop_params,
                                                expected_RR50_placebo_response_map):

    theo_trial_arm_patient_pop_monthly_means    = theo_trial_arm_patient_pop_params[:, 0]
    theo_trial_arm_patient_pop_monthly_std_devs = theo_trial_arm_patient_pop_params[:, 1]

    [theo_trial_arm_patient_pop_hist, _, _] = np.histogram2d(theo_trial_arm_patient_pop_monthly_means, 
                                                             theo_trial_arm_patient_pop_monthly_std_devs, 
                                                             bins=[16, 16], 
                                                             range=[[0, 16], [0, 16]])
    theo_trial_arm_patient_pop_hist = np.flipud(np.transpose(theo_trial_arm_patient_pop_hist))
    theo_trial_arm_patient_pop_hist = theo_trial_arm_patient_pop_hist/np.sum(np.nansum(theo_trial_arm_patient_pop_hist, 0))
    
    expected_RR50_trial_arm_response = np.sum(np.nansum(np.multiply(expected_RR50_placebo_response_map, theo_trial_arm_patient_pop_hist)))

    return expected_RR50_trial_arm_response


def calculate_map_based_statistical_power(expected_RR50_placebo_arm_response,
                                          expected_RR50_drug_arm_response,
                                          num_theo_patients_per_trial_arm):

    command = ['Rscript', 'python_and_R_scripts/Fisher_Exact_Power_Calc.R', str(expected_RR50_placebo_arm_response), str(expected_RR50_drug_arm_response), str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    fisher_exact_stat_power = float(process.communicate()[0].decode().split()[1])

    return fisher_exact_stat_power


def estimate_power_of_theo_pop(maps_folder,
                               monthly_mean_min,
                               monthly_mean_max,
                               monthly_std_dev_min,
                               monthly_std_dev_max,
                               theo_placebo_arm_patient_pop_params,
                               theo_drug_arm_patient_pop_params,
                               num_theo_patients_per_trial_arm):

    [expected_RR50_placebo_response_map, 
     expected_RR50_drug_response_map    ] = \
         get_RR50_response_maps(maps_folder)

    expected_RR50_placebo_arm_response = \
        predict_RR50_placebo_and_drug_arm_responses(monthly_mean_min,
                                                    monthly_mean_max,
                                                    monthly_std_dev_min,
                                                    monthly_std_dev_max,
                                                    theo_placebo_arm_patient_pop_params,
                                                expected_RR50_placebo_response_map)


    expected_RR50_drug_arm_response = \
        predict_RR50_placebo_and_drug_arm_responses(monthly_mean_min,
                                                    monthly_mean_max,
                                                    monthly_std_dev_min,
                                                    monthly_std_dev_max,
                                                    theo_drug_arm_patient_pop_params,
                                                expected_RR50_drug_response_map)
    
    fisher_exact_stat_power = \
        calculate_map_based_statistical_power(expected_RR50_placebo_arm_response,
                                              expected_RR50_drug_arm_response,
                                              num_theo_patients_per_trial_arm)
    
    return fisher_exact_stat_power

