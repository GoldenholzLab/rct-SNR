import numpy as np
import json
import pandas as pd
import os
import subprocess


def retrieve_map_location_data(folder, 
                               monthly_mean, 
                               monthly_std_dev, 
                               trial_arm):

    if((trial_arm == 'placebo_arm') or (trial_arm == 'drug_arm')):

        file_name_prefix = 'mean_' + str(np.int_(monthly_mean)) + '_std_' + str(np.int_(monthly_std_dev)) + '_' + trial_arm

        TTP_times_file_name        = folder + '/' + file_name_prefix + '_TTP_times.json'
        TTP_observations_file_name = folder + '/' + file_name_prefix + '_TTP_observations.json'

        with open(TTP_times_file_name, 'r') as json_file:

            TTP_times = np.array(json.load(json_file))
    
        with open(TTP_observations_file_name, 'r') as json_file:

            TTP_observations = np.array(json.load(json_file))
    
        return [TTP_times, TTP_observations]
    
    else:

        raise ValueError('The \'trial_arm\' variable must either be \'placebo_arm\' or \'drug_arm\'')


def collect_population_data(folder,
                            trial_arm,
                            num_theo_patients_per_trial_arm,
                            theo_patient_pop_params):

    trial_arm_TTP_times        = np.array([])
    trial_arm_TTP_observations = np.array([])

    monthly_mean_array    = theo_patient_pop_params[:, 0]
    monthly_std_dev_array = theo_patient_pop_params[:, 1]

    for patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = monthly_mean_array[patient_index]
        monthly_std_dev = monthly_std_dev_array[patient_index]

        [tmp_trial_arm_TTP_times, 
         tmp_trial_arm_TTP_observations] = \
             retrieve_map_location_data(folder, 
                                        monthly_mean, 
                                        monthly_std_dev, 
                                        trial_arm)
        
        trial_arm_TTP_times        = np.concatenate([trial_arm_TTP_times,        tmp_trial_arm_TTP_times])
        trial_arm_TTP_observations = np.concatenate([trial_arm_TTP_observations, tmp_trial_arm_TTP_observations])
    
    return [trial_arm_TTP_times, trial_arm_TTP_observations]


def calculate_one_theo_pop_analytical_quantities(placebo_arm_TTP_times,
                                                 placebo_arm_TTP_observations,
                                                 drug_arm_TTP_times,
                                                 drug_arm_TTP_observations,
                                                 num_theo_patients_per_trial_arm,
                                                 num_patients_per_map_location,
                                                 tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_arm_TTP_times, drug_arm_TTP_times)
    events                 = np.append(placebo_arm_TTP_observations, drug_arm_TTP_observations)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*num_patients_per_map_location*['C']), np.array(num_theo_patients_per_trial_arm*num_patients_per_map_location*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'python_and_R_scripts/estimate_log_hazard_ratio.R', relative_tmp_file_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])

    os.remove(relative_tmp_file_path)

    prob_fail_placebo_arm = np.sum(placebo_arm_TTP_observations == True)/(num_theo_patients_per_trial_arm*num_patients_per_map_location)
    prob_fail_drug_arm    = np.sum(drug_arm_TTP_observations == True)/(num_theo_patients_per_trial_arm*num_patients_per_map_location)

    return [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm]


def estimate_map_based_statistical_power(num_theo_patients_per_trial_arm,
                                         prob_fail_drug_arm,
                                         prob_fail_placebo_arm,
                                         postulated_log_hazard_ratio,
                                         alpha):

    command        = ['Rscript', 'python_and_R_scripts/calculate_cph_power.R', str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm), 
                      str(prob_fail_drug_arm), str(prob_fail_placebo_arm), str(np.exp(postulated_log_hazard_ratio)), str(alpha)]
    process        = subprocess.Popen(command, stdout=subprocess.PIPE)
    map_stat_power = float(process.communicate()[0].decode().split()[1])

    return map_stat_power


def estimate_map_based_stat_power_for_one_pop(folder,
                                              num_theo_patients_per_trial_arm,
                                              num_patients_per_map_location,
                                              tmp_file_name,
                                              alpha,
                                              theo_placebo_arm_patient_pop_params,
                                              theo_drug_arm_patient_pop_params):

    [placebo_arm_TTP_times, 
     placebo_arm_TTP_observations] = \
         collect_population_data(folder,
                                 'placebo_arm',
                                 num_theo_patients_per_trial_arm,
                                 theo_placebo_arm_patient_pop_params)
    
    [drug_arm_TTP_times, 
     drug_arm_TTP_observations] = \
         collect_population_data(folder,
                                 'drug_arm',
                                 num_theo_patients_per_trial_arm,
                                 theo_drug_arm_patient_pop_params)

    [postulated_log_hazard_ratio, 
     prob_fail_placebo_arm, 
     prob_fail_drug_arm] = \
        calculate_one_theo_pop_analytical_quantities(placebo_arm_TTP_times,
                                                     placebo_arm_TTP_observations,
                                                     drug_arm_TTP_times,
                                                     drug_arm_TTP_observations,
                                                     num_theo_patients_per_trial_arm,
                                                     num_patients_per_map_location,
                                                     tmp_file_name)
    
    map_stat_power = \
        estimate_map_based_statistical_power(num_theo_patients_per_trial_arm,
                                             prob_fail_drug_arm,
                                             prob_fail_placebo_arm,
                                             postulated_log_hazard_ratio,
                                             alpha)
    
    return map_stat_power

