import numpy as np
from .seizure_diary_generation import generate_placebo_arm_seizure_diary
from .seizure_diary_generation import generate_drug_arm_seizure_diary


def generate_theo_patient_pop_params(monthly_mean_min,
                                     monthly_mean_max,
                                     monthly_std_dev_min,
                                     monthly_std_dev_max,
                                     num_theo_patients_per_trial_arm):

    theo_patient_pop_params = np.zeros((num_theo_patients_per_trial_arm, 2))

    for patient_index in range(num_theo_patients_per_trial_arm):

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

        theo_patient_pop_params[patient_index, 0] = monthly_mean
        theo_patient_pop_params[patient_index, 1] = monthly_std_dev
    
    return theo_patient_pop_params


def convert_theo_pop_hist(monthly_mean_min,
                          monthly_mean_max,
                          monthly_std_dev_min,
                          monthly_std_dev_max,
                          theo_trial_arm_patient_pop_params):

    trial_arm_monthly_means    = theo_trial_arm_patient_pop_params[:, 0]
    trial_arm_monthly_std_devs = theo_trial_arm_patient_pop_params[:, 1]

    num_monthly_mean_bins    = monthly_mean_max    - (monthly_mean_min    - 1)
    num_monthly_std_dev_bins = monthly_std_dev_max - (monthly_std_dev_min - 1)

    hist_bins = [num_monthly_std_dev_bins, num_monthly_mean_bins]
    hist_range = [[monthly_std_dev_min, num_monthly_std_dev_bins + monthly_std_dev_min], [monthly_mean_min, num_monthly_mean_bins + monthly_mean_min]]

    [theo_trial_arm_pop_hist,_,_] = np.histogram2d(trial_arm_monthly_std_devs, trial_arm_monthly_means, bins=hist_bins, range=hist_range)
    theo_trial_arm_pop_hist = np.flipud(theo_trial_arm_pop_hist)

    return theo_trial_arm_pop_hist


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

    if(monthly_mean == 0):

        raise ValueError('The true monthly mean of a homogenous patient population cannot be zero.')

    if(monthly_std_dev <= np.sqrt(monthly_mean)):

        raise ValueError('The monthly standard deviation must be greater than the square root of the monthly mean for a homogenous patient population.')

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

    if(monthly_mean == 0):

        raise ValueError('The true monthly mean of a homogenous patient population cannot be zero.')

    if(monthly_std_dev <= np.sqrt(monthly_mean)):

        raise ValueError('The monthly standard deviation must be greater than the square root of the monthly mean for a homogenous patient population.')

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

