import numpy as np
from seizure_diary_generation import generate_placebo_arm_seizure_diary
from seizure_diary_generation import generate_drug_arm_seizure_diary


def generate_theo_patient_pop_params(monthly_mean_min,
                                     monthly_mean_max,
                                     monthly_std_dev_min,
                                     monthly_std_dev_max,
                                     num_theo_patients_per_trial_arm):

    theo_patient_pop_params = np.zeros((num_theo_patients_per_trial_arm, 2))

    for patient_index in range(num_theo_patients_per_trial_arm):

        overdispersed = False
        
        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max    + 1)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max + 1)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        theo_patient_pop_params[patient_index, 0] = monthly_mean
        theo_patient_pop_params[patient_index, 1] = monthly_std_dev
    
    return theo_patient_pop_params


def generate_homogenous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                monthly_mean, 
                                                monthly_std_dev,
                                                num_baseline_months,
                                                num_testing_months,
                                                baseline_time_scaling_const,
                                                testing_time_scaling_const,
                                                minimum_required_baseline_seizure_count,
                                                placebo_mu,
                                                placebo_sigma):

    num_baseline_scaled_time_units = num_baseline_months*baseline_time_scaling_const
    num_testing_scaled_time_units  = num_testing_months*testing_time_scaling_const

    placebo_arm_baseline_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_baseline_scaled_time_units))
    placebo_arm_testing_seizure_diaries  = np.zeros((num_theo_patients_per_trial_arm, num_testing_scaled_time_units))

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        [placebo_arm_baseline_seizure_diary, placebo_arm_testing_seizure_diary] = \
            generate_placebo_arm_seizure_diary(monthly_mean, 
                                               monthly_std_dev,
                                               num_baseline_months,
                                               num_testing_months,
                                               baseline_time_scaling_const,
                                               testing_time_scaling_const,
                                               minimum_required_baseline_seizure_count,
                                               placebo_mu,
                                               placebo_sigma)
        
        placebo_arm_baseline_seizure_diaries[theo_patient_index, :] = placebo_arm_baseline_seizure_diary
        placebo_arm_testing_seizure_diaries[theo_patient_index, :]  = placebo_arm_testing_seizure_diary

    return [placebo_arm_baseline_seizure_diaries, 
            placebo_arm_testing_seizure_diaries  ]


def generate_homogenous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
                                             monthly_mean, 
                                             monthly_std_dev,
                                             num_baseline_months,
                                             num_testing_months,
                                             baseline_time_scaling_const,
                                             testing_time_scaling_const,
                                             minimum_required_baseline_seizure_count,
                                             placebo_mu,
                                             placebo_sigma,
                                             drug_mu,
                                             drug_sigma):

    num_baseline_scaled_time_units = num_baseline_months*baseline_time_scaling_const
    num_testing_scaled_time_units  = num_testing_months*testing_time_scaling_const

    drug_arm_baseline_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_baseline_scaled_time_units))
    drug_arm_testing_seizure_diaries  = np.zeros((num_theo_patients_per_trial_arm, num_testing_scaled_time_units))

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        [drug_arm_baseline_seizure_diary, drug_arm_testing_seizure_diary] = \
            generate_drug_arm_seizure_diary(monthly_mean, 
                                               monthly_std_dev,
                                               num_baseline_months,
                                               num_testing_months,
                                               baseline_time_scaling_const,
                                               testing_time_scaling_const,
                                               minimum_required_baseline_seizure_count,
                                               placebo_mu,
                                               placebo_sigma,
                                               drug_mu,
                                               drug_sigma)
        
        drug_arm_baseline_seizure_diaries[theo_patient_index, :] = drug_arm_baseline_seizure_diary
        drug_arm_testing_seizure_diaries[theo_patient_index, :]  = drug_arm_testing_seizure_diary

    return [drug_arm_baseline_seizure_diaries, 
            drug_arm_testing_seizure_diaries  ]


def generate_heterogenous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                  theo_placebo_arm_patient_pop_params,
                                                  num_baseline_months,
                                                  num_testing_months,
                                                  baseline_time_scaling_const,
                                                  testing_time_scaling_const,
                                                  minimum_required_baseline_seizure_count,
                                                  placebo_mu,
                                                  placebo_sigma):

    num_baseline_scaled_time_units = num_baseline_months*baseline_time_scaling_const
    num_testing_scaled_time_units  = num_testing_months*testing_time_scaling_const

    placebo_arm_baseline_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_baseline_scaled_time_units))
    placebo_arm_testing_seizure_diaries  = np.zeros((num_theo_patients_per_trial_arm, num_testing_scaled_time_units))

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = theo_placebo_arm_patient_pop_params[theo_patient_index, 0]
        monthly_std_dev = theo_placebo_arm_patient_pop_params[theo_patient_index, 1]

        [placebo_arm_baseline_seizure_diary, placebo_arm_testing_seizure_diary] = \
            generate_placebo_arm_seizure_diary(monthly_mean, 
                                               monthly_std_dev,
                                               num_baseline_months,
                                               num_testing_months,
                                               baseline_time_scaling_const,
                                               testing_time_scaling_const,
                                               minimum_required_baseline_seizure_count,
                                               placebo_mu,
                                               placebo_sigma)
        
        placebo_arm_baseline_seizure_diaries[theo_patient_index, :] = placebo_arm_baseline_seizure_diary
        placebo_arm_testing_seizure_diaries[theo_patient_index, :]  = placebo_arm_testing_seizure_diary

    return [placebo_arm_baseline_seizure_diaries, 
            placebo_arm_testing_seizure_diaries  ]


def generate_heterogenous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
                                               theo_drug_arm_patient_pop_params,
                                               num_baseline_months,
                                               num_testing_months,
                                               baseline_time_scaling_const,
                                               testing_time_scaling_const,
                                               minimum_required_baseline_seizure_count,
                                               placebo_mu,
                                               placebo_sigma,
                                               drug_mu,
                                               drug_sigma):

    num_baseline_scaled_time_units = num_baseline_months*baseline_time_scaling_const
    num_testing_scaled_time_units  = num_testing_months*testing_time_scaling_const

    drug_arm_baseline_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_baseline_scaled_time_units))
    drug_arm_testing_seizure_diaries  = np.zeros((num_theo_patients_per_trial_arm, num_testing_scaled_time_units))

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = theo_drug_arm_patient_pop_params[theo_patient_index, 0]
        monthly_std_dev = theo_drug_arm_patient_pop_params[theo_patient_index, 1]

        [drug_arm_baseline_seizure_diary, drug_arm_testing_seizure_diary] = \
            generate_drug_arm_seizure_diary(monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            num_testing_months,
                                            baseline_time_scaling_const,
                                            testing_time_scaling_const,
                                            minimum_required_baseline_seizure_count,
                                            placebo_mu,
                                            placebo_sigma,
                                            drug_mu,
                                            drug_sigma)
        
        drug_arm_baseline_seizure_diaries[theo_patient_index, :] = drug_arm_baseline_seizure_diary
        drug_arm_testing_seizure_diaries[theo_patient_index, :]  = drug_arm_testing_seizure_diary

    return [drug_arm_baseline_seizure_diaries, 
            drug_arm_testing_seizure_diaries  ]

'''
if(__name__=='__main__'):

    monthly_mean_min    = 4
    monthly_mean_max    = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 8
    num_theo_patients_per_trial_arm = 153

    num_baseline_months = 2
    num_testing_months = 3
    baseline_time_scaling_const = 28
    testing_time_scaling_const = 28
    minimum_required_baseline_seizure_count = 4
    placebo_mu = 0
    placebo_sigma = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05

    theo_placebo_arm_patient_pop = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)
    
    theo_drug_arm_patient_pop = \
        generate_theo_patient_pop_params(monthly_mean_min,
                                         monthly_mean_max,
                                         monthly_std_dev_min,
                                         monthly_std_dev_max,
                                         num_theo_patients_per_trial_arm)

    [placebo_arm_baseline_seizure_diaries, 
     placebo_arm_testing_seizure_diaries  ] = \
         generate_heterogenous_placebo_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                       theo_placebo_arm_patient_pop,
                                                       num_baseline_months,
                                                       num_testing_months,
                                                       baseline_time_scaling_const,
                                                       testing_time_scaling_const,
                                                       minimum_required_baseline_seizure_count,
                                                       placebo_mu,
                                                       placebo_sigma)
    
    [drug_arm_baseline_seizure_diaries, 
     drug_arm_testing_seizure_diaries  ] = \
         generate_heterogenous_drug_arm_patient_pop(num_theo_patients_per_trial_arm,
                                                    theo_drug_arm_patient_pop,
                                                    num_baseline_months,
                                                    num_testing_months,
                                                    baseline_time_scaling_const,
                                                    testing_time_scaling_const,
                                                    minimum_required_baseline_seizure_count,
                                                    placebo_mu,
                                                    placebo_sigma,
                                                    drug_mu,
                                                    drug_sigma)

    import pandas as pd
    print('\n' + pd.DataFrame(np.transpose(theo_placebo_arm_patient_pop)).to_string() + '\n')
    print('\n' + pd.DataFrame(placebo_arm_baseline_seizure_diaries).to_string() + '\n')
    print('\n' + pd.DataFrame(placebo_arm_testing_seizure_diaries).to_string()  + '\n')
    print('\n' + pd.DataFrame(np.transpose(theo_drug_arm_patient_pop)).to_string() + '\n')
    print('\n' + pd.DataFrame(drug_arm_baseline_seizure_diaries).to_string()    + '\n')
    print('\n' + pd.DataFrame(drug_arm_testing_seizure_diaries).to_string()     + '\n')
'''