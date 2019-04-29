import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd


def generate_counts(shape, scale, alpha, beta, drug_effect,
                    num_base_intervals, num_test_intervals, num_patients, num_days_per_interval):
    
    num_total_intervals = num_base_intervals + num_test_intervals
    
    interval_counts = np.zeros((num_patients, num_total_intervals))
    
    patient_index = 0
    
    while(patient_index < num_patients):
        
        n = np.random.gamma(shape, 1/scale)
        p = np.random.beta(alpha, beta)

        mu = n*(1 - p)/p
        alpha = 1/n
        
        for interval in range(num_total_intervals):
            
            rate = np.random.gamma(num_days_per_interval/alpha, mu*alpha)
            count = np.random.poisson(rate)
            
            if(interval >= num_base_intervals):
            
                num_removed = 0
            
                for seizure in range(count):
            
                    if(np.random.uniform(0, 1) < drug_effect):
                
                        num_removed = num_removed + 1
                        
                count = count - num_removed
            
            interval_counts[patient_index, interval] = count
        
        if( np.sum(interval_counts[patient_index, 0:num_base_intervals]) >= 4 ):
            
            patient_index = patient_index + 1

    return interval_counts


def calculate_percent_changes_and_endpoints(shape, scale, alpha, beta, drug_effect,
                                            num_base_intervals, num_test_intervals, num_patients, num_days_per_interval):

    interval_counts = generate_counts(shape, scale, alpha, beta, drug_effect,
                                        num_base_intervals, num_test_intervals, num_patients, num_days_per_interval)

    base_interval_counts = interval_counts[:, 0:num_base_weeks]
    test_interval_counts = interval_counts[:, num_base_weeks:]

    base_interval_count_averages = np.mean(base_interval_counts, 1)
    test_interval_count_averages = np.mean(test_interval_counts, 1)

    percent_changes = np.divide(base_interval_count_averages - test_interval_count_averages, base_interval_count_averages)

    RR50 = 100*np.sum(percent_changes >= 0.5)/num_patients
    MPC = 100*np.median(percent_changes)

    return [RR50, MPC, percent_changes]


def generate_trial(shape, scale, alpha, beta, drug_effect,
                   num_base_intervals, num_test_intervals, num_patients, num_days_per_interval):
    
    [placebo_RR50, placebo_MPC, _] = \
            calculate_percent_changes_and_endpoints(shape, scale, alpha, beta, placebo_effect, 
                                                    num_base_weeks, num_test_weeks, num_patients_per_arm, num_days_per_interval)

    [drug_RR50, drug_MPC, _] = \
            calculate_percent_changes_and_endpoints(shape, scale, alpha, beta, drug_effect, 
                                                    num_base_weeks, num_test_weeks, num_patients_per_arm, num_days_per_interval)
    
    return [placebo_RR50, drug_RR50, placebo_MPC, drug_MPC]

def calculate_endpoint_statistics(shape, scale, alpha, beta,  drug_effect, 
                                   num_base_intervals, num_test_intervals, num_patients_per_arm, num_days_per_interval):
    
    placebo_RR50_array = np.zeros(num_trials)
    drug_RR50_array = np.zeros(num_trials)
    placebo_MPC_array = np.zeros(num_trials)
    drug_MPC_array = np.zeros(num_trials)
    
    for i in range(num_trials):

        [placebo_RR50, drug_RR50, placebo_MPC, drug_MPC] = \
                    generate_trial(shape, scale, alpha, beta, drug_effect, 
                                   num_base_intervals, num_test_intervals, num_patients_per_arm, num_days_per_interval)
                    
        placebo_RR50_array[i] = placebo_RR50
        drug_RR50_array[i] = drug_RR50
        placebo_MPC_array[i] = placebo_MPC
        drug_MPC_array[i] = drug_MPC

        
    mean_RR50_placebo_arm = np.mean(placebo_RR50_array)
    std_RR50_placebo_arm = np.std(placebo_RR50_array)
    mean_RR50_drug_arm = np.mean(drug_RR50_array)
    std_RR50_drug_arm = np.std(drug_RR50_array)
    
    mean_MPC_placebo_arm = np.mean(placebo_MPC_array)
    std_MPC_placebo_arm = np.std(placebo_MPC_array)
    mean_MPC_drug_arm = np.mean(drug_MPC_array)
    std_MPC_drug_arm = np.std(drug_MPC_array)
    
    return [mean_RR50_placebo_arm, std_RR50_placebo_arm, mean_RR50_drug_arm, std_RR50_drug_arm, \
            mean_MPC_placebo_arm, std_MPC_placebo_arm, mean_MPC_drug_arm, std_MPC_drug_arm]

if(__name__ == '__main__'):
    
    # NV model parameters
    '''
    shape= 24.143
    scale = 297.366
    alpha = 284.366
    beta = 369.628
    '''
    shape = 111.313
    scale = 296.728
    alpha = 296.339
    beta = 243.719

    # placebo and drug effect parameters
    drug_effect = 0.2
    placebo_effect = 0

    # simulate RCT trials
    num_base_weeks = 8
    num_test_weeks = 12
    num_patients_per_arm = 153
    num_days_per_week = 7
    num_trials = 500

    [mean_RR50_placebo_arm, std_RR50_placebo_arm, mean_RR50_drug_arm, std_RR50_drug_arm, \
     mean_MPC_placebo_arm,  std_MPC_placebo_arm,  mean_MPC_drug_arm,  std_MPC_drug_arm] = \
         calculate_endpoint_statistics(shape, scale, alpha, beta, placebo_effect + drug_effect, 
                                       num_base_weeks, num_test_weeks, num_patients_per_arm, num_days_per_week)

    print('\n\nRR50 placebo arm: ' + str(mean_RR50_placebo_arm) + ' +- ' + str(std_RR50_placebo_arm)
            + '\nRR50 drug arm: ' + str(mean_RR50_drug_arm) + ' +- ' + str(std_RR50_drug_arm) +
            '\n\nMPC placebo arm: ' + str(mean_MPC_placebo_arm) + ' +- ' + str(std_MPC_placebo_arm)
            + '\nMPC drug arm: ' + str(mean_MPC_drug_arm) + ' +- ' + str(std_MPC_drug_arm)  + '\n\n')

