import glob
import json
import numpy as np


def calculate_num_patients_and_costs(folder_name,
                                     num_baseline_months,
                                     num_testing_months):

    num_patients_RR50_list        = []
    num_patients_MPC_list         = []
    num_patients_placebo_TTP_list = []
    num_patients_drug_TTP_list    = []
    average_placebo_TTP_list      = []
    average_drug_TTP_list         = []
    
    for file_name_str in glob.glob(folder_name + '/*.json'):

        with open(file_name_str, 'r') as json_file:
            
            iter_data = json.load(json_file)
            num_patients_RR50_list.append(iter_data[0])
            num_patients_MPC_list.append(iter_data[1])
            num_patients_placebo_TTP_list.append(iter_data[2])
            num_patients_drug_TTP_list.append(iter_data[3])
            average_placebo_TTP_list.append(iter_data[4])
            average_drug_TTP_list.append(iter_data[5])
    
    num_patients_RR50_array        = np.array(num_patients_RR50_list)
    num_patients_MPC_array         = np.array(num_patients_MPC_list)
    num_patients_placebo_TTP_array = np.array(num_patients_placebo_TTP_list)
    num_patients_drug_TTP_array    = np.array(num_patients_drug_TTP_list)
    average_placebo_TTP_array      = np.array(average_placebo_TTP_list)
    average_drug_TTP_array         = np.array(average_drug_TTP_list)

    average_num_patients_RR50 = np.mean(num_patients_RR50_array)
    average_num_patients_MPC  = np.mean(num_patients_MPC_array)
    average_num_patients_TTP  = np.mean(num_patients_placebo_TTP_array + num_patients_drug_TTP_array)

    std_dev_num_patients_RR50 = np.std(num_patients_RR50_array)
    std_dev_num_patients_MPC  = np.std(num_patients_MPC_array)
    std_dev_num_patients_TTP  = np.std(num_patients_placebo_TTP_array + num_patients_drug_TTP_array)

    RR50_cost = average_num_patients_RR50*(num_baseline_months + num_testing_months)*1295
    MPC_cost  =  average_num_patients_MPC*(num_baseline_months + num_testing_months)*1295

    placebo_TTP_time = np.round(np.mean( num_patients_placebo_TTP_array ))*( num_baseline_months + np.mean( average_placebo_TTP_array)/28 )
    drug_TTP_time    = np.round(np.mean(    num_patients_drug_TTP_array ))*( num_testing_months  + np.mean(   average_drug_TTP_array)/28  )
    TTP_cost = (placebo_TTP_time + drug_TTP_time)*1295

    return [average_num_patients_RR50, 
            average_num_patients_MPC, 
            average_num_patients_TTP,
            std_dev_num_patients_RR50,
            std_dev_num_patients_MPC,
            std_dev_num_patients_TTP,
            RR50_cost,
            MPC_cost,
            TTP_cost]


def text_output(dumb_num_patients_RR50,
                dumb_num_patients_MPC,
                dumb_num_patients_TTP,
                smart_num_patients_RR50,
                smart_num_patients_MPC,
                smart_num_patients_TTP,
                dumb_num_patients_RR50_range,
                dumb_num_patients_MPC_range,
                dumb_num_patients_TTP_range,
                smart_num_patients_RR50_range,
                smart_num_patients_MPC_range,
                smart_num_patients_TTP_range,
                dumb_RR50_cost,
                dumb_MPC_cost,
                dumb_TTP_cost,
                smart_RR50_cost,
                smart_MPC_cost,
                smart_TTP_cost):

    dumb_num_patients_RR50_str  = str(dumb_num_patients_RR50)
    dumb_num_patients_MPC_str   = str(dumb_num_patients_MPC)
    dumb_num_patients_TTP_str   = str(dumb_num_patients_TTP)
    smart_num_patients_RR50_str = str(smart_num_patients_RR50)
    smart_num_patients_MPC_str  = str(smart_num_patients_MPC)
    smart_num_patients_TTP_str  = str(smart_num_patients_TTP)

    dumb_num_patients_RR50_range_str  = str(dumb_num_patients_RR50_range)
    dumb_num_patients_MPC_range_str   = str(dumb_num_patients_MPC_range)
    dumb_num_patients_TTP_range_str   = str(dumb_num_patients_TTP_range)
    smart_num_patients_RR50_range_str = str(smart_num_patients_RR50_range)
    smart_num_patients_MPC_range_str  = str(smart_num_patients_MPC_range)
    smart_num_patients_TTP_range_str  = str(smart_num_patients_TTP_range)

    dumb_RR50_cost_str  = str(np.int_(np.round(dumb_RR50_cost)))
    dumb_MPC_cost_str   = str(np.int_(np.round(dumb_MPC_cost)))
    dumb_TTP_cost_str   = str(np.int_(np.round(dumb_TTP_cost)))
    smart_RR50_cost_str = str(np.int_(np.round(smart_RR50_cost)))
    smart_MPC_cost_str  = str(np.int_(np.round(smart_MPC_cost)))
    smart_TTP_cost_str  = str(np.int_(np.round(smart_TTP_cost)))
    RR50_savings_str    = str(np.int_(np.round(dumb_RR50_cost - smart_RR50_cost)))
    MPC_savings_str     = str(np.int_(np.round(dumb_MPC_cost  - smart_MPC_cost)))
    TTP_savings_str     = str(np.int_(np.round(dumb_TTP_cost  - smart_TTP_cost)))

    num_patients_col_str = '           RR50      MPC      TTP'
    dumb_num_patients_str  = 'dumb:   ' + dumb_num_patients_RR50_str  + ' ± ' + dumb_num_patients_RR50_range_str + \
                                  ',  ' + dumb_num_patients_MPC_str   + ' ± ' + dumb_num_patients_MPC_range_str  + \
                                  ',  ' + dumb_num_patients_TTP_str   + ' ± ' + dumb_num_patients_TTP_range_str
    smart_num_patients_str = 'smart:  ' + smart_num_patients_RR50_str + ' ± ' + smart_num_patients_RR50_range_str + \
                                  ',  ' + smart_num_patients_MPC_str  + ' ± ' + smart_num_patients_MPC_range_str  + \
                                  ',  ' + smart_num_patients_TTP_str  + ' ± ' + smart_num_patients_TTP_range_str

    costs_col_str = '                RR50        MPC       TTP'
    dumb_costs_str  = 'dumb costs:   $'  + dumb_RR50_cost_str  + ',   $' + dumb_MPC_cost_str  + ',   $' + dumb_TTP_cost_str
    smart_costs_str = 'smart costs:  $'  + smart_RR50_cost_str + ',   $' + smart_MPC_cost_str + ',   $' + smart_TTP_cost_str
    savings_str     = 'savings:      $'  + RR50_savings_str    + ',   $' + MPC_savings_str    + ',    $' + TTP_savings_str

    output_str = '\n'   + num_patients_col_str  + '\n\n' + dumb_num_patients_str + '\n\n' + smart_num_patients_str + '\n\n' + \
                 '\n\n' + costs_col_str         + '\n\n' + dumb_costs_str        + '\n\n' + smart_costs_str        + '\n\n' + \
                          savings_str           + '\n'

    print(output_str)


if(__name__=='__main__'):

    num_baseline_months = 2
    num_testing_months  = 3

    [average_dumb_num_patients_RR50, 
     average_dumb_num_patients_MPC, 
     average_dumb_num_patients_TTP,
     dumb_std_dev_num_patients_RR50,
     dumb_std_dev_num_patients_MPC,
     dumb_std_dev_num_patients_TTP,
     dumb_RR50_cost,
     dumb_MPC_cost,
     dumb_TTP_cost] = \
         calculate_num_patients_and_costs('dumb_data',
                                          num_baseline_months,
                                          num_testing_months)
    
    [average_smart_num_patients_RR50, 
     average_smart_num_patients_MPC, 
     average_smart_num_patients_TTP,
     smart_std_dev_num_patients_RR50,
     smart_std_dev_num_patients_MPC,
     smart_std_dev_num_patients_TTP,
     smart_RR50_cost,
     smart_MPC_cost,
     smart_TTP_cost] = \
         calculate_num_patients_and_costs('smart_data',
                                          num_baseline_months,
                                          num_testing_months)
    
    dumb_num_patients_RR50  = np.int_(np.round(average_dumb_num_patients_RR50))
    dumb_num_patients_MPC   = np.int_(np.round(average_dumb_num_patients_MPC))
    dumb_num_patients_TTP   = np.int_(np.round(average_dumb_num_patients_TTP))
    smart_num_patients_RR50 = np.int_(np.round(average_smart_num_patients_RR50))
    smart_num_patients_MPC  = np.int_(np.round(average_smart_num_patients_MPC))
    smart_num_patients_TTP  = np.int_(np.round(average_smart_num_patients_TTP))

    dumb_num_patients_RR50_range  = np.int_(np.round(dumb_std_dev_num_patients_RR50))
    dumb_num_patients_MPC_range   = np.int_(np.round(dumb_std_dev_num_patients_MPC))
    dumb_num_patients_TTP_range   = np.int_(np.round(dumb_std_dev_num_patients_RR50))
    smart_num_patients_RR50_range = np.int_(np.round(smart_std_dev_num_patients_RR50))
    smart_num_patients_MPC_range  = np.int_(np.round(smart_std_dev_num_patients_MPC))
    smart_num_patients_TTP_range  = np.int_(np.round(smart_std_dev_num_patients_RR50))


    text_output(dumb_num_patients_RR50,
                dumb_num_patients_MPC,
                dumb_num_patients_TTP,
                smart_num_patients_RR50,
                smart_num_patients_MPC,
                smart_num_patients_TTP,
                dumb_num_patients_RR50_range,
                dumb_num_patients_MPC_range,
                dumb_num_patients_TTP_range,
                smart_num_patients_RR50_range,
                smart_num_patients_MPC_range,
                smart_num_patients_TTP_range,
                dumb_RR50_cost,
                dumb_MPC_cost,
                dumb_TTP_cost,
                smart_RR50_cost,
                smart_MPC_cost,
                smart_TTP_cost)
    
