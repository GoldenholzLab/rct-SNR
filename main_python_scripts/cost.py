import sys
import os
import json
import numpy as np
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary

def retrieve_SNR_map(endpoint_name):

    with open(endpoint_name + '_SNR_data.json', 'r') as json_file:

        SNR_map = np.array(json.load(json_file))
    
    return SNR_map


def get_patient_loc(monthly_mean_min,
                    monthly_mean_max,
                    monthly_std_dev_min,
                    monthly_std_dev_max):

    overdispersed = False
    non_zero_mean = False
    realistic = False

    while((not overdispersed) or (not non_zero_mean) or (not realistic)):

        overdispersed = False
        non_zero_mean = False
        realistic = False
        
        monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max    + 1)
        monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max + 1)

        if(monthly_mean != 0):

                non_zero_mean = True

        if(monthly_std_dev > np.sqrt(monthly_mean)):

            overdispersed = True
        
        if(monthly_std_dev < np.power(monthly_mean, 1.3)):

            realistic = True
    
    return [monthly_mean, monthly_std_dev]


def estimate_patient_loc(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_baseline_months):

    '''

    I suspect that standard deviation is conitually underestimated.

    '''

    estims_within_SNR_map = False

    while(estims_within_SNR_map == False):

        [monthly_mean, monthly_std_dev] = \
            get_patient_loc(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max)

        baseline_monthly_seizure_diary = \
            generate_seizure_diary(num_baseline_months, 
                                   monthly_mean, 
                                   monthly_std_dev, 
                                   1)
    
        monthly_mean_hat    = np.round(np.mean(baseline_monthly_seizure_diary))
        monthly_std_dev_hat = np.round(np.std(baseline_monthly_seizure_diary))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )
        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map
    
    return [monthly_mean_hat, monthly_std_dev_hat]


if(__name__=='__main__'):

    endpoint_name = sys.argv[1]

    SNR_map = retrieve_SNR_map(endpoint_name)

    monthly_mean_min    = 1
    monthly_mean_max    = 15
    monthly_std_dev_min = 1
    monthly_std_dev_max = 15

    num_baseline_months = 2

    [monthly_mean_hat, monthly_std_dev_hat] = \
        estimate_patient_loc(monthly_mean_min,
                             monthly_mean_max,
                             monthly_std_dev_min,
                             monthly_std_dev_max,
                             num_baseline_months)

    # have to take care of estimated underdispersion
    print(monthly_std_dev_hat > np.sqrt(monthly_mean_hat))

    #SNR_hat
    #import pandas as pd
    #print(pd.DataFrame(SNR_map).to_string())

