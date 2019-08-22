import numpy as np
import os
import json
import subprocess
import pandas as pd


def retrieve_map_location_TTP_times(folder, monthly_mean, monthly_std_dev, trial_arm_str):

    file_name = str(monthly_mean) + '_' + str(monthly_std_dev) + '_' + trial_arm_str
    file_path = folder + '/' + file_name + '.json'
    with open(file_path, 'r') as json_file:
        TTP_times = np.array(json.load(json_file))

    return TTP_times


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


def retrieve_theo_patient_pop_TTP_times(monthly_mean_min,
                                        monthly_mean_max, 
                                        monthly_std_dev_min, 
                                        monthly_std_dev_max, 
                                        num_theo_patients_per_trial_arm,
                                        folder,
                                        trial_arm_str):

    patient_pop_monthly_param_set = \
        generate_patient_pop_params(monthly_mean_min,
                                    monthly_mean_max, 
                                    monthly_std_dev_min, 
                                    monthly_std_dev_max, 
                                    num_theo_patients_per_trial_arm)
    
    trial_arm_TTP_times = np.array([])

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = int(patient_pop_monthly_param_set[theo_patient_index, 0])
        monthly_std_dev = int(patient_pop_monthly_param_set[theo_patient_index, 1])

        tmp_trial_arm_TTP_times = retrieve_map_location_TTP_times(folder, monthly_mean, monthly_std_dev, trial_arm_str)

        trial_arm_TTP_times = np.concatenate([trial_arm_TTP_times, tmp_trial_arm_TTP_times])
    
    num_TTP_times = len(trial_arm_TTP_times)

    trial_arm_TTP_observed_array = np.zeros(num_TTP_times)
    trial_arm_TTP_observed_array[ trial_arm_TTP_times == 84 ] = 1

    return [trial_arm_TTP_times, trial_arm_TTP_observed_array]


def generate_one_trial_analytical_quantities(placebo_arm_TTP_times,
                                             placebo_arm_TTP_observed_array,
                                             drug_arm_TTP_times,
                                             drug_arm_TTP_observed_array,
                                             num_theo_patients_per_trial_arm,
                                             num_patients_per_dot,
                                             tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_arm_TTP_times, drug_arm_TTP_times)
    events                 = np.append(placebo_arm_TTP_observed_array, drug_arm_TTP_observed_array)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*num_patients_per_dot*['C']) , np.array(num_theo_patients_per_trial_arm*num_patients_per_dot*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'estimate_log_hazard_ratio.R', relative_tmp_file_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])

    os.remove(relative_tmp_file_path)

    prob_fail_placebo_arm = np.sum(placebo_arm_TTP_observed_array == True)/(num_theo_patients_per_trial_arm*num_patients_per_dot)
    prob_fail_drug_arm    = np.sum(drug_arm_TTP_observed_array == True)/(num_theo_patients_per_trial_arm*num_patients_per_dot)

    return [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm]


if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8

    num_theo_patients_per_trial_arm = 1
    num_patients_per_dot            = 5000

    folder = os.getcwd() + '/hist_maps_folder'

    placebo_arm_str = 'placebo'
    drug_arm_str    = 'drug'

    [placebo_arm_TTP_times, placebo_arm_TTP_observed_array] = \
        retrieve_theo_patient_pop_TTP_times(monthly_mean_min,
                                            monthly_mean_max, 
                                            monthly_std_dev_min, 
                                            monthly_std_dev_max, 
                                            num_theo_patients_per_trial_arm,
                                            folder,
                                            placebo_arm_str)
    
    [drug_arm_TTP_times, drug_arm_TTP_observed_array] = \
        retrieve_theo_patient_pop_TTP_times(monthly_mean_min,
                                            monthly_mean_max, 
                                            monthly_std_dev_min, 
                                            monthly_std_dev_max, 
                                            num_theo_patients_per_trial_arm,
                                            folder,
                                            drug_arm_str)

    tmp_file_name = 'tmp'

    [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm] = \
        generate_one_trial_analytical_quantities(placebo_arm_TTP_times,
                                                 placebo_arm_TTP_observed_array,
                                                 drug_arm_TTP_times,
                                                 drug_arm_TTP_observed_array,
                                                 num_theo_patients_per_trial_arm,
                                                 num_patients_per_dot,
                                                 tmp_file_name)
    
    print(np.round(np.array([postulated_log_hazard_ratio, 100*prob_fail_placebo_arm, 100*prob_fail_drug_arm]), 6))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist()
    plt.figure()
    plt.hist()
    plt.show()
