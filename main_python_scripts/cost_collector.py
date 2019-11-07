import sys
import glob
import numpy as np


def retrive_num_patients_from_RR50_or_MPC(folder_name):

    file_name_str_array = glob.glob(folder_name + '/*.txt')

    num_patients_list = []

    for file_name_str in file_name_str_array:
        
        with open(file_name_str , 'r') as text_file:

            num_patients_list.append(np.int_(text_file.read()))

    num_patients_array = np.array(num_patients_list)

    return num_patients_array


def retrive_num_patients_from_TTP(folder_name):

    file_name_str_array = glob.glob(folder_name + '/*.txt')

    num_patients_list = []
    expected_placebo_arm_TTP_list = []
    expected_drug_arm_TTP_list = []

    for file_name_str in file_name_str_array:
        
        with open(file_name_str , 'r') as text_file:

            str_array = text_file.read().split(',')
            num_patients_list.append(np.int_(str_array[0]))
            expected_placebo_arm_TTP_list.append(np.float(str_array[1]))
            expected_drug_arm_TTP_list.append(np.float(str_array[2]))
        
    num_patients_array = np.array(num_patients_list)
    expected_placebo_arm_TTP_array = np.array(expected_placebo_arm_TTP_list)
    expected_drug_arm_TTP_array = np.array(expected_drug_arm_TTP_list)

    return [num_patients_array, expected_placebo_arm_TTP_array, expected_drug_arm_TTP_array]


def calculate_cost_of_RR50_or_MPC(endpoint_name,
                                  num_baseline_months,
                                  num_testing_months):

    endpoint_name_smart_folder_name = endpoint_name + '_smart'
    endpoint_name_dumb_folder_name  = endpoint_name + '_dumb'

    smart_num_patients_array = retrive_num_patients_from_RR50_or_MPC(endpoint_name_smart_folder_name)
    dumb_num_patients_array  = retrive_num_patients_from_RR50_or_MPC(endpoint_name_dumb_folder_name)
    
    smart_num_patients = np.int_(np.round(np.mean(smart_num_patients_array)))
    dumb_num_patients  = np.int_(np.round(np.mean(dumb_num_patients_array)))

    smart_cost = smart_num_patients*(num_baseline_months + num_testing_months)*1295
    dumb_cost  =  dumb_num_patients*(num_baseline_months + num_testing_months)*1295

    return [smart_num_patients, dumb_num_patients, smart_cost, dumb_cost]


def calculate_cost_of_TTP(endpoint_name, num_baseline_months):

    endpoint_name_smart_folder_name = endpoint_name + '_smart'
    endpoint_name_dumb_folder_name  = endpoint_name + '_dumb'

    [smart_num_patients_array, 
     smart_expected_placebo_arm_TTP_array, 
     smart_expected_drug_arm_TTP_array] = \
         retrive_num_patients_from_TTP(endpoint_name_smart_folder_name)

    [dumb_num_patients_array, 
     dumb_expected_placebo_arm_TTP_array, 
     dumb_expected_drug_arm_TTP_array] = \
         retrive_num_patients_from_TTP(endpoint_name_dumb_folder_name)
    
    smart_num_patients = np.int_(np.round(np.mean(smart_num_patients_array)))
    dumb_num_patients  = np.int_(np.round(np.mean(dumb_num_patients_array)))

    smart_num_patients_per_trial_arm = np.int_(np.round(np.mean(smart_num_patients_array)/2))
    dumb_num_patients_per_trial_arm  = np.int_(np.round(np.mean(dumb_num_patients_array)/2))

    weighted_average_smart_expected_placebo_arm_TTP = np.dot(smart_num_patients_array/np.sum(smart_num_patients_array), smart_expected_placebo_arm_TTP_array)
    weighted_average_smart_expected_drug_arm_TTP    = np.dot(smart_num_patients_array/np.sum(smart_num_patients_array), smart_expected_drug_arm_TTP_array)
    weighted_average_dumb_expected_placebo_arm_TTP  = np.dot(dumb_num_patients_array/np.sum(dumb_num_patients_array),   dumb_expected_placebo_arm_TTP_array)
    weighted_average_dumb_expected_drug_arm_TTP     = np.dot(dumb_num_patients_array/np.sum(dumb_num_patients_array),   dumb_expected_drug_arm_TTP_array)

    smart_placebo_cost = smart_num_patients_per_trial_arm*(num_baseline_months + (weighted_average_smart_expected_placebo_arm_TTP/28))*1295
    smart_drug_cost    = smart_num_patients_per_trial_arm*(num_baseline_months + (weighted_average_smart_expected_drug_arm_TTP/28))*1295
    dumb_placebo_cost  =  dumb_num_patients_per_trial_arm*(num_baseline_months + (weighted_average_dumb_expected_placebo_arm_TTP/28))*1295
    dumb_drug_cost     =  dumb_num_patients_per_trial_arm*(num_baseline_months + (weighted_average_dumb_expected_drug_arm_TTP/28))*1295

    smart_cost = np.int_(np.round(smart_placebo_cost + smart_drug_cost))
    dumb_cost  = np.int_(np.round(dumb_placebo_cost  + dumb_drug_cost))

    return [smart_num_patients, dumb_num_patients, smart_cost, dumb_cost]


if(__name__=='__main__'):

    endpoint_name = sys.argv[1]

    num_baseline_months = 2
    num_testing_months = 3
    
    if(endpoint_name == 'RR50' or endpoint_name == 'MPC'):

        [smart_num_patients, 
         dumb_num_patients, 
         smart_cost, 
         dumb_cost] = \
             calculate_cost_of_RR50_or_MPC(endpoint_name,
                                           num_baseline_months,
                                           num_testing_months)
    
    elif(endpoint_name == 'TTP'):

        [smart_num_patients, 
         dumb_num_patients, 
         smart_cost, 
         dumb_cost] = \
             calculate_cost_of_TTP(endpoint_name, num_baseline_months)

    print(  '\nnumber of patients, smart algorithm: ' + str(smart_num_patients) + \
            '\nnumber of patients,  dumb algorithm: ' + str(dumb_num_patients)  + \
            '\nsmart algorithm cost: ' + str(smart_cost) + \
          ' $\n dumb algorithm cost: ' + str(dumb_cost)  + ' $\n')
