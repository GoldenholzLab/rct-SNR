import numpy as np
import json
import sys
import os
sys.path.insert(0, os.getcwd())
from utility_code.seizure_diary_generation import generate_seizure_diary_with_minimum_count
from utility_code.empirical_estimation import empirically_estimate_all_endpoint_statistical_powers


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

    while((not overdispersed) or (not non_zero_mean)):

        overdispersed = False
        non_zero_mean = False
        
        monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max    + 1)
        monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max + 1)

        if(monthly_mean != 0):

                non_zero_mean = True

        if(monthly_std_dev > np.sqrt(monthly_mean)):

            overdispersed = True
    
    return [monthly_mean, monthly_std_dev]


def estimate_patient_loc(monthly_mean_min,
                         monthly_mean_max,
                         monthly_std_dev_min,
                         monthly_std_dev_max,
                         num_baseline_months,
                         minimum_required_baseline_seizure_count):

    estims_within_SNR_map = False

    while(estims_within_SNR_map == False):

        [monthly_mean, monthly_std_dev] = \
            get_patient_loc(monthly_mean_min,
                            monthly_mean_max,
                            monthly_std_dev_min,
                            monthly_std_dev_max)

        baseline_monthly_seizure_diary = \
            generate_seizure_diary_with_minimum_count(num_baseline_months, 
                                                      monthly_mean, 
                                                      monthly_std_dev, 
                                                      1,
                                                      minimum_required_baseline_seizure_count)
    
        monthly_mean_hat    = np.int_(np.round(np.mean(baseline_monthly_seizure_diary)))
        monthly_std_dev_hat = np.int_(np.round(np.std(baseline_monthly_seizure_diary)))

        monthly_mean_hat_within_SNR_map    = (    monthly_mean_min < monthly_mean_hat    ) and (    monthly_mean_hat < monthly_mean_max    )
        monthly_std_dev_hat_within_SNR_map = ( monthly_std_dev_min < monthly_std_dev_hat ) and ( monthly_std_dev_hat < monthly_std_dev_max )

        estimated_location_overdispersed = monthly_std_dev_hat > np.sqrt(monthly_mean_hat)

        estims_within_SNR_map = monthly_mean_hat_within_SNR_map and monthly_std_dev_hat_within_SNR_map and estimated_location_overdispersed
    
    return [monthly_mean_hat, monthly_std_dev_hat]

'''
def calculate_percent_change_p_values(num_theo_patients_per_placebo_arm,
                                      placebo_arm_baseline_monthly_seizure_diaries,
                                      placebo_arm_testing_monthly_seizure_diaries,
                                      num_theo_patients_per_drug_arm,
                                      drug_arm_baseline_monthly_seizure_diaries,
                                      drug_arm_testing_monthly_seizure_diaries,
                                      update_RR50,
                                      update_MPC):

    RR50_p_value = 1
    MPC_p_value  = 1

    placebo_arm_percent_changes = \
        calculate_percent_changes(placebo_arm_baseline_monthly_seizure_diaries,
                                  placebo_arm_testing_monthly_seizure_diaries,
                                  num_theo_patients_per_placebo_arm)
    
    drug_arm_percent_changes = \
        calculate_percent_changes(drug_arm_baseline_monthly_seizure_diaries,
                                  drug_arm_testing_monthly_seizure_diaries,
                                  num_theo_patients_per_drug_arm)

    if( update_RR50 and not update_MPC ):

        RR50_p_value = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)

        return RR50_p_value
        
    elif( not update_RR50 and update_MPC ):

        MPC_p_value = \
            calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                             drug_arm_percent_changes)
        
        return MPC_p_value

    elif( update_RR50 and update_MPC ):

        RR50_p_value = \
            calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                           drug_arm_percent_changes)
        
        MPC_p_value = \
            calculate_Mann_Whitney_U_p_value(placebo_arm_percent_changes,
                                             drug_arm_percent_changes)
    
        return [RR50_p_value, MPC_p_value]


def calculate_prerandomization_p_value(num_theo_patients_per_placebo_arm,
                                       placebo_arm_baseline_monthly_seizure_diaries,
                                       placebo_arm_testing_daily_seizure_diaries,
                                       num_theo_patients_per_drug_arm,
                                       drug_arm_baseline_monthly_seizure_diaries,
                                       drug_arm_testing_daily_seizure_diaries,
                                       num_testing_days):
    
    [placebo_arm_TTP_times, placebo_arm_observed_array] = \
        calculate_time_to_prerandomizations(placebo_arm_baseline_monthly_seizure_diaries,
                                            placebo_arm_testing_daily_seizure_diaries,
                                            num_theo_patients_per_placebo_arm,
                                            num_testing_days)
    
    [drug_arm_TTP_times, drug_arm_observed_array] = \
        calculate_time_to_prerandomizations(drug_arm_baseline_monthly_seizure_diaries,
                                            drug_arm_testing_daily_seizure_diaries,
                                            num_theo_patients_per_drug_arm,
                                            num_testing_days)
        
    TTP_p_value = \
        calculate_logrank_p_value(placebo_arm_TTP_times, 
                                  placebo_arm_observed_array, 
                                  drug_arm_TTP_times, 
                                  drug_arm_observed_array)
    
    return TTP_p_value


def efficiently_generate_p_values(num_theo_patients_per_placebo_arm,
                                  num_theo_patients_per_drug_arm,
                                  theo_placebo_arm_patient_pop_params,
                                  theo_drug_arm_patient_pop_params,
                                  num_baseline_months,
                                  num_testing_months,
                                  minimum_required_baseline_seizure_count,
                                  placebo_mu,
                                  placebo_sigma,
                                  drug_mu,
                                  drug_sigma,
                                  update_RR50,
                                  update_MPC,
                                  update_TTP):
    
    baseline_time_scaling_const = 1
    RR50_p_value = 1 
    MPC_p_value  = 1
    TTP_p_value  = 1

    if( update_TTP ):

        testing_time_scaling_const = 28
        num_testing_days = num_testing_months*testing_time_scaling_const

        [placebo_arm_baseline_monthly_seizure_diaries, 
         placebo_arm_testing_daily_seizure_diaries  ] = \
             generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_placebo_arm,
                                                            theo_placebo_arm_patient_pop_params,
                                                            num_baseline_months,
                                                            num_testing_months,
                                                            baseline_time_scaling_const,
                                                            testing_time_scaling_const,
                                                            minimum_required_baseline_seizure_count,
                                                            placebo_mu,
                                                            placebo_sigma)

        [drug_arm_baseline_monthly_seizure_diaries, 
         drug_arm_testing_daily_seizure_diaries  ] = \
             generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_per_drug_arm,
                                                         theo_drug_arm_patient_pop_params,
                                                         num_baseline_months,
                                                         num_testing_months,
                                                         baseline_time_scaling_const,
                                                         testing_time_scaling_const,
                                                         minimum_required_baseline_seizure_count,
                                                         placebo_mu,
                                                         placebo_sigma,
                                                         drug_mu,
                                                         drug_sigma)
        
        TTP_p_value = \
            calculate_prerandomization_p_value(num_theo_patients_per_placebo_arm,
                                               placebo_arm_baseline_monthly_seizure_diaries,
                                               placebo_arm_testing_daily_seizure_diaries,
                                               num_theo_patients_per_drug_arm,
                                               drug_arm_baseline_monthly_seizure_diaries,
                                               drug_arm_testing_daily_seizure_diaries,
                                               num_testing_days)

        if( update_RR50 or update_MPC ):

            placebo_arm_testing_monthly_seizure_diaries = \
                np.sum(placebo_arm_testing_daily_seizure_diaries.reshape((num_theo_patients_per_placebo_arm,
                                                                          num_testing_months,
                                                                          testing_time_scaling_const)), 2)
    
            drug_arm_testing_monthly_seizure_diaries = \
                np.sum(drug_arm_testing_daily_seizure_diaries.reshape((num_theo_patients_per_drug_arm,
                                                                       num_testing_months, 
                                                                       testing_time_scaling_const)), 2)

            [placebo_arm_baseline_monthly_seizure_diaries, 
             placebo_arm_testing_monthly_seizure_diaries  ] = \
                 generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_placebo_arm,
                                                                theo_placebo_arm_patient_pop_params,
                                                                num_baseline_months,
                                                                num_testing_months,
                                                                baseline_time_scaling_const,
                                                                testing_time_scaling_const,
                                                                minimum_required_baseline_seizure_count,
                                                                placebo_mu,
                                                                placebo_sigma)

            [drug_arm_baseline_monthly_seizure_diaries, 
             drug_arm_testing_monthly_seizure_diaries  ] = \
                 generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_per_drug_arm,
                                                             theo_drug_arm_patient_pop_params,
                                                             num_baseline_months,
                                                             num_testing_months,
                                                             baseline_time_scaling_const,
                                                             testing_time_scaling_const,
                                                             minimum_required_baseline_seizure_count,
                                                             placebo_mu,
                                                             placebo_sigma,
                                                             drug_mu,
                                                             drug_sigma)
        
            [RR50_p_value, 
             MPC_p_value] = \
                 calculate_percent_change_p_values(num_theo_patients_per_placebo_arm,
                                                   placebo_arm_baseline_monthly_seizure_diaries,
                                                   placebo_arm_testing_monthly_seizure_diaries,
                                                   num_theo_patients_per_drug_arm,
                                                   drug_arm_baseline_monthly_seizure_diaries,
                                                   drug_arm_testing_monthly_seizure_diaries,
                                                   update_RR50,
                                                   update_MPC)
            
            return [RR50_p_value, MPC_p_value, TTP_p_value]
        
        else:

            return TTP_p_value

    elif( update_RR50 or update_MPC ):

        testing_time_scaling_const  = 1

        [placebo_arm_baseline_monthly_seizure_diaries, 
         placebo_arm_testing_monthly_seizure_diaries  ] = \
             generate_heterogeneous_placebo_arm_patient_pop(num_theo_patients_per_placebo_arm,
                                                            theo_placebo_arm_patient_pop_params,
                                                            num_baseline_months,
                                                            num_testing_months,
                                                            baseline_time_scaling_const,
                                                            testing_time_scaling_const,
                                                            minimum_required_baseline_seizure_count,
                                                            placebo_mu,
                                                            placebo_sigma)

        [drug_arm_baseline_monthly_seizure_diaries, 
         drug_arm_testing_monthly_seizure_diaries  ] = \
             generate_heterogeneous_drug_arm_patient_pop(num_theo_patients_per_drug_arm,
                                                         theo_drug_arm_patient_pop_params,
                                                         num_baseline_months,
                                                         num_testing_months,
                                                         baseline_time_scaling_const,
                                                         testing_time_scaling_const,
                                                         minimum_required_baseline_seizure_count,
                                                         placebo_mu,
                                                         placebo_sigma,
                                                         drug_mu,
                                                         drug_sigma)
        
        [RR50_p_value, 
         MPC_p_value] = \
             calculate_percent_change_p_values(num_theo_patients_per_placebo_arm,
                                              placebo_arm_baseline_monthly_seizure_diaries,
                                              placebo_arm_testing_monthly_seizure_diaries,
                                              num_theo_patients_per_drug_arm,
                                              drug_arm_baseline_monthly_seizure_diaries,
                                              drug_arm_testing_monthly_seizure_diaries,
                                              update_RR50,
                                              update_MPC)
        
        return [RR50_p_value, MPC_p_value]

    else:

        raise ValueError('Something went wrong in the drug cost algorithm')



def calculate_trial_endpoints(num_testing_months,
                              num_theo_patients_per_placebo_arm,
                              num_theo_patients_per_drug_arm,
                              placebo_arm_baseline_monthly_seizure_diaries,
                              placebo_arm_testing_monthly_seizure_diaries,
                              placebo_arm_testing_daily_seizure_diaries,
                              drug_arm_baseline_monthly_seizure_diaries,
                              drug_arm_testing_monthly_seizure_diaries,
                              drug_arm_testing_daily_seizure_diaries,
                              calculate_percent_changes,
                              calculate_prerandomization_times):

    if( calculate_percent_changes ):
        placebo_arm_percent_changes = \
                calculate_percent_changes(placebo_arm_baseline_monthly_seizure_diaries,
                                          placebo_arm_testing_monthly_seizure_diaries,
                                          num_theo_patients_per_placebo_arm)
    
        drug_arm_percent_changes = \
                calculate_percent_changes(drug_arm_baseline_monthly_seizure_diaries,
                                          drug_arm_testing_monthly_seizure_diaries,
                                          num_theo_patients_per_drug_arm)

    if( calculate_prerandomization_times ):

        num_testing_days = num_testing_months*28

        [placebo_arm_TTP_times, placebo_arm_observed_array] = \
                calculate_time_to_prerandomizations(placebo_arm_baseline_monthly_seizure_diaries,
                                                    placebo_arm_testing_daily_seizure_diaries,
                                                    num_theo_patients_per_placebo_arm,
                                                    num_testing_days)
    
        [drug_arm_TTP_times, drug_arm_observed_array] = \
                calculate_time_to_prerandomizations(drug_arm_baseline_monthly_seizure_diaries,
                                                    drug_arm_testing_daily_seizure_diaries,
                                                    num_theo_patients_per_drug_arm,
                                                    num_testing_days)
    
    if( calculate_percent_changes and not calculate_prerandomization_times ):

        return [placebo_arm_percent_changes,
                drug_arm_percent_changes]

    elif( not calculate_percent_changes and calculate_prerandomization_times ):
    
        return [placebo_arm_TTP_times, 
                placebo_arm_observed_array,
                drug_arm_TTP_times, 
                drug_arm_observed_array]
    
    elif(calculate_percent_changes and calculate_prerandomization_times):

        return [placebo_arm_percent_changes,
                drug_arm_percent_changes,
                placebo_arm_TTP_times, 
                placebo_arm_observed_array,
                drug_arm_TTP_times, 
                drug_arm_observed_array]
    
    else:

        raise ValueError('Something went wrong with the empirically guided cost algorithm.')


def empirically_estimate_all_endpoint_statistical_powers(num_theo_patients_per_placebo_arm,
                                                         num_theo_patients_per_drug_arm,
                                                         theo_placebo_arm_patient_pop_params,
                                                         theo_drug_arm_patient_pop_params,
                                                         num_baseline_months,
                                                         num_testing_months,
                                                         minimum_required_baseline_seizure_count,
                                                         placebo_mu,
                                                         placebo_sigma,
                                                         drug_mu,
                                                         drug_sigma,
                                                         num_trials,
                                                         update_RR50,
                                                         update_MPC,
                                                         update_TTP):

    RR50_p_value_array = np.ones(num_trials)
    MPC_p_value_array  = np.ones(num_trials)
    TTP_p_value_array  = np.ones(num_trials)

    for trial_index in range(num_trials):
    
        if(update_TTP):

            if(update_RR50):
            
            if(update_MPC):

        [RR50_p_value_array[trial_index], 
         MPC_p_value_array[trial_index], 
         TTP_p_value_array[trial_index]] = \
             efficiently_generate_p_values(num_theo_patients_per_placebo_arm,
                                           num_theo_patients_per_drug_arm,
                                           theo_placebo_arm_patient_pop_params,
                                           theo_drug_arm_patient_pop_params,
                                           num_baseline_months,
                                           num_testing_months,
                                           minimum_required_baseline_seizure_count,
                                           placebo_mu,
                                           placebo_sigma,
                                           drug_mu,
                                           drug_sigma,
                                           update_RR50,
                                           update_MPC,
                                           update_TTP)
    
    RR50_stat_power = np.sum(RR50_p_value_array <= 0.05)/num_trials
    MPC_stat_power  = np.sum(MPC_p_value_array  <= 0.05)/num_trials
    TTP_stat_power  = np.sum(TTP_p_value_array  <= 0.05)/num_trials

    return [RR50_stat_power, MPC_stat_power, TTP_stat_power]
'''



if(__name__=='__main__'):

    monthly_mean_min    = 1
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16

    num_baseline_months = 2
    num_testing_months = 3
    minimum_required_baseline_seizure_count = 4

    placebo_mu    = 0
    placebo_sigma = 0.05
    drug_mu       = 0.2
    drug_sigma    = 0.05

    num_trials_per_stat_power_estim = 10
    target_stat_power  = 0.5
    num_extra_patients = 20

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    
    theo_placebo_arm_patient_pop_params_hat_list = []
    theo_drug_arm_patient_pop_params_hat_list    = []

    theo_placebo_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                             monthly_mean_max,
                                                                             monthly_std_dev_min,
                                                                             monthly_std_dev_max,
                                                                             num_baseline_months,
                                                                             minimum_required_baseline_seizure_count))

    theo_drug_arm_patient_pop_params_hat_list.append(estimate_patient_loc(monthly_mean_min,
                                                                          monthly_mean_max,
                                                                          monthly_std_dev_min,
                                                                          monthly_std_dev_max,
                                                                          num_baseline_months,
                                                                          minimum_required_baseline_seizure_count))

    num_theo_patients_per_placebo_arm = 1
    num_theo_patients_per_drug_arm = 1
    RR50_stat_power = 0
    MPC_stat_power  = 0
    TTP_stat_power  = 0
    current_trial_arm = 'placebo'

    num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm
    

    update_RR50 = True
    update_MPC  = True
    update_TTP  = True
    underpowered_endpoints = True

    num_RR50_patients = num_patients
    num_MPC_patients  = num_patients
    num_TTP_patients  = num_patients

    while(underpowered_endpoints):

        [monthly_mean_hat, 
         monthly_std_dev_hat] = \
             estimate_patient_loc(monthly_mean_min,
                                  monthly_mean_max,
                                  monthly_std_dev_min,
                                  monthly_std_dev_max,
                                  num_baseline_months,
                                  minimum_required_baseline_seizure_count)

        if( current_trial_arm == 'placebo' ):
            theo_placebo_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            num_theo_patients_per_placebo_arm = num_theo_patients_per_placebo_arm + 1
            current_trial_arm = 'drug' 

        elif( current_trial_arm == 'drug' ):
            theo_drug_arm_patient_pop_params_hat_list.append([monthly_mean_hat, monthly_std_dev_hat])
            num_theo_patients_per_drug_arm = num_theo_patients_per_drug_arm + 1
            current_trial_arm = 'placebo'

        num_patients = num_theo_patients_per_placebo_arm + num_theo_patients_per_drug_arm

        [RR50_stat_power, 
         MPC_stat_power, 
         TTP_stat_power] = \
             empirically_estimate_all_endpoint_statistical_powers(num_theo_patients_per_placebo_arm,
                                                                  num_theo_patients_per_drug_arm,
                                                                  np.array(theo_placebo_arm_patient_pop_params_hat_list),
                                                                  np.array(theo_drug_arm_patient_pop_params_hat_list),
                                                                  num_baseline_months,
                                                                  num_testing_months,
                                                                  minimum_required_baseline_seizure_count,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  drug_mu,
                                                                  drug_sigma,
                                                                  num_trials_per_stat_power_estim)
        
        underpowered_endpoints = update_RR50 or update_MPC or update_TTP

        if( update_RR50 ):
            if( RR50_stat_power < target_stat_power ):
                num_RR50_patients = num_patients
            else:
                update_RR50 = False
        if( update_MPC ):
            if( MPC_stat_power  < target_stat_power ):
                num_MPC_patients = num_patients
            else:
                update_MPC = False
        if( update_TTP ):
            if( TTP_stat_power  < target_stat_power ):
                num_TTP_patients = num_patients
            else:
                update_TTP = False

        print()
        print(100*np.array([RR50_stat_power,   MPC_stat_power,   TTP_stat_power]))
        print([num_RR50_patients, num_MPC_patients, num_TTP_patients])
        #print( str(theo_placebo_arm_patient_pop_params_hat_list) + '\n' + str(theo_drug_arm_patient_pop_params_hat_list) )
        print()

    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------#

