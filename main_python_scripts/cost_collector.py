import glob
import json
import numpy as np

def RR50_costs(smart_or_dumb,
               num_baseline_months,
               num_testing_months):

    num_patients_RR50_list        = []
    RR50_folder_name = 'RR50_' + smart_or_dumb + '_data'
    for file_name_str in glob.glob(RR50_folder_name + '/*.json'):
        with open(file_name_str, 'r') as json_file:
            iter_data = json.load(json_file)
            num_patients_RR50_list.append(iter_data)

    num_patients_RR50_array   = np.array(num_patients_RR50_list)
    average_num_patients_RR50 = np.mean(num_patients_RR50_array)
    std_dev_num_patients_RR50 = np.std(num_patients_RR50_array)/np.sqrt(len(num_patients_RR50_list))
    RR50_cost = average_num_patients_RR50*(num_baseline_months + num_testing_months)*1295

    return [average_num_patients_RR50,
            std_dev_num_patients_RR50,
            RR50_cost]


def MPC_costs(smart_or_dumb,
               num_baseline_months,
               num_testing_months):

    MPC_folder_name  = 'MPC_'  + smart_or_dumb + '_data'
    num_patients_MPC_list         = []
    for file_name_str in glob.glob(MPC_folder_name + '/*.json'):
        with open(file_name_str, 'r') as json_file:
            iter_data = json.load(json_file)
            num_patients_MPC_list.append(iter_data)

    num_patients_MPC_array   = np.array(num_patients_MPC_list)
    average_num_patients_MPC = np.mean(num_patients_MPC_array)
    std_dev_num_patients_MPC = np.std(num_patients_MPC_array)/np.sqrt(len(num_patients_MPC_list))
    MPC_cost = average_num_patients_MPC*(num_baseline_months + num_testing_months)*1295

    return [average_num_patients_MPC,
            std_dev_num_patients_MPC,
            MPC_cost]


def TTP_costs(smart_or_dumb,
               num_baseline_months,
               num_testing_months):

    TTP_folder_name  = 'TTP_'  + smart_or_dumb + '_data'
    num_patients_placebo_TTP_list = []
    num_patients_drug_TTP_list    = []
    average_placebo_TTP_list      = []
    average_drug_TTP_list         = []
    for file_name_str in glob.glob(TTP_folder_name + '/*.json'):
        with open(file_name_str, 'r') as json_file:
            iter_data = json.load(json_file)
            num_patients_placebo_TTP_list.append(iter_data[0])
            num_patients_drug_TTP_list.append(iter_data[1])
            average_placebo_TTP_list.append(iter_data[2])
            average_drug_TTP_list.append(iter_data[3])
    
    num_patients_placebo_TTP_array = np.array(num_patients_placebo_TTP_list)
    num_patients_drug_TTP_array    = np.array(num_patients_drug_TTP_list)
    average_placebo_TTP_array      = np.array(average_placebo_TTP_list)
    average_drug_TTP_array         = np.array(average_drug_TTP_list)
    average_num_patients_TTP  = np.mean(num_patients_placebo_TTP_array + num_patients_drug_TTP_array)
    std_dev_num_patients_TTP  = np.std(num_patients_placebo_TTP_array + num_patients_drug_TTP_array)/np.sqrt(len(num_patients_placebo_TTP_array) + len(num_patients_drug_TTP_array))
    placebo_TTP_time = np.round(np.mean( num_patients_placebo_TTP_array ))*( num_baseline_months + np.mean( average_placebo_TTP_array)/28 )
    drug_TTP_time    = np.round(np.mean(    num_patients_drug_TTP_array ))*( num_testing_months  + np.mean(   average_drug_TTP_array)/28  )
    TTP_cost = (placebo_TTP_time + drug_TTP_time)*1295

    return [average_num_patients_TTP,
            std_dev_num_patients_TTP,
            TTP_cost]


if(__name__=='__main__'):

    smart_or_dumb = 'dumb'
    num_baseline_months = 2
    num_testing_months  = 3
    
    
    [dumb_average_num_patients_RR50,
     dumb_std_dev_num_patients_RR50,
     dumb_RR50_cost] = \
         RR50_costs('dumb',
                     num_baseline_months,
                     num_testing_months)
    
    [smart_average_num_patients_RR50,
     smart_std_dev_num_patients_RR50,
     smart_RR50_cost] = \
         RR50_costs('smart',
                     num_baseline_months,
                     num_testing_months)

    print(str(np.int_(np.round(dumb_average_num_patients_RR50)))  + ' ± ' + str(np.int_(np.round(dumb_std_dev_num_patients_RR50)))  + ', $ ' + str(np.int_(np.round(dumb_RR50_cost))))
    print(str(np.int_(np.round(smart_average_num_patients_RR50))) + ' ± ' + str(np.int_(np.round(smart_std_dev_num_patients_RR50))) + ', $ ' + str(np.int_(np.round(smart_RR50_cost))))
    

    '''
    [dumb_average_num_patients_MPC,
     dumb_std_dev_num_patients_MPC,
     dumb_MPC_cost] = \
         MPC_costs('dumb',
                   num_baseline_months,
                   num_testing_months)
    
    [smart_average_num_patients_MPC,
     smart_std_dev_num_patients_MPC,
     smart_MPC_cost] = \
         MPC_costs('smart',
                   num_baseline_months,
                   num_testing_months)

    print(str(np.int_(np.round(dumb_average_num_patients_MPC)))  + ' ± ' + str(np.int_(np.round(dumb_std_dev_num_patients_MPC)))  + ', $ ' + str(np.int_(np.round(dumb_MPC_cost))))
    print(str(np.int_(np.round(smart_average_num_patients_MPC))) + ' ± ' + str(np.int_(np.round(smart_std_dev_num_patients_MPC))) + ', $ ' + str(np.int_(np.round(smart_MPC_cost))))
    '''

    '''
    [dumb_average_num_patients_TTP,
     dumb_std_dev_num_patients_TTP,
     dumb_TTP_cost] = \
         TTP_costs('dumb',
                   num_baseline_months,
                   num_testing_months)
    
    [smart_average_num_patients_TTP,
     smart_std_dev_num_patients_TTP,
     smart_TTP_cost] = \
         TTP_costs('smart',
                   num_baseline_months,
                   num_testing_months)

    print(str(np.int_(np.round(dumb_average_num_patients_TTP)))  + ' ± ' + str(np.int_(np.round(dumb_std_dev_num_patients_TTP)))  + ', $ ' + str(np.int_(np.round(dumb_TTP_cost))))
    print(str(np.int_(np.round(smart_average_num_patients_TTP))) + ' ± ' + str(np.int_(np.round(smart_std_dev_num_patients_TTP))) + ', $ ' + str(np.int_(np.round(smart_TTP_cost))))
    '''

