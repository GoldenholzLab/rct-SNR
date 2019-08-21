import numpy as np
import json
import os
import subprocess
import pandas as pd


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


def generate_one_patient_pop_trial_arm_hist(median_TTP_time_map_file_name, 
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
    
    median_TTP_times = np.zeros(num_theo_patients_per_trial_arm)

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = int(patient_pop_monthly_param_sets[theo_patient_index, 0])
        monthly_std_dev = int(patient_pop_monthly_param_sets[theo_patient_index, 1])
    
        median_TTP_times[theo_patient_index] = median_TTP_time_map[16 - monthly_std_dev, monthly_mean - 1]
    
    median_TTP_observed_array = np.zeros(num_theo_patients_per_trial_arm)
    median_TTP_observed_array[ median_TTP_times == 84 ] = 1
    
    return [median_TTP_times, median_TTP_observed_array]


def estimate_analytical_patient_pop_quantities(placebo_arm_median_TTP_times, 
                                               placebo_arm_median_TTP_observed_array,
                                               drug_arm_median_TTP_times, 
                                               drug_arm_median_TTP_observed_array,
                                               num_theo_patients_per_trial_arm,
                                               tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_arm_median_TTP_times, drug_arm_median_TTP_times)
    events                 = np.append(placebo_arm_median_TTP_observed_array, drug_arm_median_TTP_observed_array)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*['C']) , np.array(num_theo_patients_per_trial_arm*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'estimate_log_hazard_ratio.R', relative_tmp_file_path]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])

    os.remove(relative_tmp_file_path)

    prob_fail_placebo_arm = np.sum(placebo_arm_median_TTP_observed_array == True)/num_theo_patients_per_trial_arm
    prob_fail_drug_arm    = np.sum(drug_arm_median_TTP_observed_array == True)/num_theo_patients_per_trial_arm

    return [postulated_log_hazard_ratio, prob_fail_placebo_arm, prob_fail_drug_arm]


if(__name__=='__main__'):

    placebo_arm_median_TTP_time_map_file_name = 'placebo_arm_median_TTP_time_map'
    drug_arm_median_TTP_time_map_file_name    =    'drug_arm_median_TTP_time_map'
    folder = os.getcwd()

    monthly_mean_min     = 4
    monthly_mean_max     = 16
    monthly_std_dev_min  = 1
    monthly_std_dev_max  = 8

    num_theo_patients_per_trial_arm = 153
    alpha = 0.05

    tmp_file_name='smaller'

    [placebo_arm_median_TTP_times, 
     placebo_arm_median_TTP_observed_array] = \
        generate_one_patient_pop_trial_arm_hist(placebo_arm_median_TTP_time_map_file_name, 
                                                folder,
                                                monthly_mean_min,
                                                monthly_mean_max, 
                                                monthly_std_dev_min, 
                                                monthly_std_dev_max, 
                                                num_theo_patients_per_trial_arm)
    
    [drug_arm_median_TTP_times, 
     drug_arm_median_TTP_observed_array] = \
        generate_one_patient_pop_trial_arm_hist(drug_arm_median_TTP_time_map_file_name, 
                                                folder,
                                                monthly_mean_min,
                                                monthly_mean_max, 
                                                monthly_std_dev_min, 
                                                monthly_std_dev_max, 
                                                num_theo_patients_per_trial_arm)

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_arm_median_TTP_times, drug_arm_median_TTP_times)
    events                 = np.append(placebo_arm_median_TTP_observed_array, drug_arm_median_TTP_observed_array)
    treatment_arms_str     = np.append( np.array(num_theo_patients_per_trial_arm*['C']) , np.array(num_theo_patients_per_trial_arm*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    command = ['Rscript', 'estimate_log_hazard_ratio.R', relative_tmp_file_path]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    print(process.communicate())
    #print( pd.DataFrame( np.vstack((placebo_arm_median_TTP_times, placebo_arm_median_TTP_observed_array, drug_arm_median_TTP_times, drug_arm_median_TTP_observed_array)).transpose() ).to_string() )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(placebo_arm_median_TTP_times, bins=25, range=[20, 84])
    plt.title('Placebo Arm')
    plt.xlabel('TTP Times')

    plt.figure()
    plt.hist(drug_arm_median_TTP_times, bins=25, range=[20, 84])
    plt.title('Drug Arm')
    plt.xlabel('TTP Times')

    plt.show()

    '''
    [postulated_log_hazard_ratio, 
     prob_fail_placebo_arm, 
     prob_fail_drug_arm          ] = \
        estimate_analytical_patient_pop_quantities(placebo_arm_median_TTP_times, 
                                               placebo_arm_median_TTP_observed_array,
                                               drug_arm_median_TTP_times, 
                                               drug_arm_median_TTP_observed_array,
                                               num_theo_patients_per_trial_arm,
                                               tmp_file_name)
    
    command = ['Rscript', 'calculate_cph_power.R', str(num_theo_patients_per_trial_arm), str(num_theo_patients_per_trial_arm), 
                      str(prob_fail_drug_arm), str(prob_fail_placebo_arm), str(np.exp(postulated_log_hazard_ratio)), str(alpha)]
    process        = subprocess.Popen(command, stdout=subprocess.PIPE)
    ana_stat_power = 100*float(process.communicate()[0].decode().split()[1])
    
    print( np.array([100*(np.exp(postulated_log_hazard_ratio) - 1), 100*prob_fail_placebo_arm, 100*prob_fail_drug_arm]) )
    print(ana_stat_power)
    '''
