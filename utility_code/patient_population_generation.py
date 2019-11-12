import numpy as np
from .seizure_diary_generation import generate_baseline_seizure_diary
from .seizure_diary_generation import generate_placebo_arm_testing_seizure_diary
from .seizure_diary_generation import generate_drug_arm_testing_seizure_diary


def randomly_select_theo_patient_pop(monthly_mean_lower_bound,
                                     monthly_mean_upper_bound,
                                     monthly_std_dev_lower_bound,
                                     monthly_std_dev_upper_bound):

    monthly_mean_min    = 0
    monthly_mean_max    = 0
    monthly_std_dev_min = 0
    monthly_std_dev_max = 0

    monthly_mean_axis_makes_sense    =    monthly_mean_min < monthly_mean_max
    monthly_std_dev_axis_makes_sense = monthly_std_dev_min < monthly_std_dev_max
    overdispersed_patients_allowed   = monthly_std_dev_max > np.sqrt(monthly_mean_min)
    patient_pop_window_makes_sense = monthly_mean_axis_makes_sense and monthly_std_dev_axis_makes_sense and overdispersed_patients_allowed

    while(not patient_pop_window_makes_sense):

        monthly_mean_min = np.random.randint(monthly_mean_lower_bound,     monthly_mean_upper_bound    )
        monthly_mean_max = np.random.randint(monthly_mean_lower_bound + 1, monthly_mean_upper_bound + 1)
    
        monthly_std_dev_min = np.random.randint(monthly_std_dev_lower_bound,     monthly_std_dev_upper_bound    )
        monthly_std_dev_max = np.random.randint(monthly_std_dev_lower_bound + 1, monthly_std_dev_upper_bound + 1)
    
        monthly_mean_axis_makes_sense    =    monthly_mean_min < monthly_mean_max
        monthly_std_dev_axis_makes_sense = monthly_std_dev_min < monthly_std_dev_max
        overdispersed_patients_allowed   = monthly_std_dev_max > np.sqrt(monthly_mean_min)
        patient_pop_window_makes_sense = monthly_mean_axis_makes_sense and monthly_std_dev_axis_makes_sense and overdispersed_patients_allowed
    
    return [monthly_mean_min, monthly_mean_max, monthly_std_dev_min, monthly_std_dev_max]


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


def generate_NV_model_patient_pop_params(num_theo_patients_per_trial_arm,
                                         one_or_two):

    if(one_or_two == 'one'):

        shape = 24.143
        scale = 297.366
        alpha = 284.024
        beta  = 369.628
    
    elif(one_or_two == 'two'):

        shape = 111.313
        scale = 296.728
        alpha = 296.339
        beta  = 243.719
    
    else:

        raise ValueError('the \'one_or_two\' parameter in the generate_NV_model_patient_pop() function is supposed to take one of two values: \'one\' or \'two\'')

    NV_model_patient_pop_params = np.zeros((num_theo_patients_per_trial_arm, 2))

    for patient_index in range(num_theo_patients_per_trial_arm):

        daily_n = np.random.gamma(shape, 1/scale)
        daily_p = np.random.beta(alpha, beta)

        odds_ratio = (1 - daily_p)/daily_p

        daily_mean    = daily_n*odds_ratio
        daily_std_dev = np.sqrt(daily_mean/daily_p)

        monthly_mean    = np.round(       28*daily_mean       )
        monthly_std_dev = np.round( np.sqrt(28)*daily_std_dev )

        NV_model_patient_pop_params[patient_index, 0] = monthly_mean
        NV_model_patient_pop_params[patient_index, 1] = monthly_std_dev

    return NV_model_patient_pop_params


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


def convert_theo_pop_hist_with_empty_regions(monthly_mean_lower_bound,
                                             monthly_mean_upper_bound,
                                             monthly_std_dev_lower_bound,
                                             monthly_std_dev_upper_bound,
                                             theo_trial_arm_patient_pop_params):

    trial_arm_monthly_means    = theo_trial_arm_patient_pop_params[:, 0]
    trial_arm_monthly_std_devs = theo_trial_arm_patient_pop_params[:, 1]

    num_monthly_mean_bins    = monthly_mean_upper_bound    - (monthly_mean_lower_bound    - 1)
    num_monthly_std_dev_bins = monthly_std_dev_upper_bound - (monthly_std_dev_lower_bound - 1)

    hist_bins = [num_monthly_std_dev_bins, num_monthly_mean_bins]
    hist_range = [[monthly_std_dev_lower_bound, num_monthly_std_dev_bins + monthly_std_dev_lower_bound], [monthly_mean_lower_bound, num_monthly_mean_bins + monthly_mean_lower_bound]]

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

        placebo_arm_baseline_seizure_diary = \
            generate_baseline_seizure_diary(monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            baseline_time_scaling_const,
                                            minimum_required_baseline_seizure_count)

        placebo_arm_testing_seizure_diary = \
            generate_placebo_arm_testing_seizure_diary(num_testing_months, 
                                                       monthly_mean, 
                                                       monthly_std_dev, 
                                                       testing_time_scaling_const,
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

        drug_arm_baseline_seizure_diary = \
            generate_baseline_seizure_diary(monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            baseline_time_scaling_const,
                                            minimum_required_baseline_seizure_count)

        drug_arm_testing_seizure_diary = \
            generate_drug_arm_testing_seizure_diary(num_testing_months, 
                                                    monthly_mean, 
                                                    monthly_std_dev, 
                                                    testing_time_scaling_const,
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

        placebo_arm_baseline_seizure_diary = \
            generate_baseline_seizure_diary(monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            baseline_time_scaling_const,
                                            minimum_required_baseline_seizure_count)
        
        placebo_arm_testing_seizure_diary = \
            generate_placebo_arm_testing_seizure_diary(num_testing_months, 
                                                       monthly_mean, 
                                                       monthly_std_dev, 
                                                       testing_time_scaling_const,
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
    '''

    Inputs:

        1) num_theo_patients_per_trial_arm:
            (int)  - 
        2) theo_drug_arm_patient_pop_params:
            (2D Numpy array)  - 
        3) num_baseline_months:
            (int) - 
        4) num_testing_months:
            (int) - 
        5) baseline_time_scaling_const:
            (int) - 
        6) testing_time_scaling_const:
            (int)  - 
        7) minimum_required_baseline_seizure_count:
            (int) - 
        8) placebo_mu:
            (float) - the mean of the normally distributed placebo effect, expressed as a percentage
        9) placebo_sigma:
            (float) - the standard deviation of the normally distributed placebo effect, expressed as 
                      a percentage
        10) drug_mu:
            (float) - the mean of the normally distributed drug effect, expressed as a percentage
        11) drug_sigma:
            (float) - the standard deviation of the normally distributed drug effect, expressed as 
                      a percentage

    '''
    
    num_baseline_scaled_time_units = num_baseline_months*baseline_time_scaling_const
    num_testing_scaled_time_units  = num_testing_months*testing_time_scaling_const

    drug_arm_baseline_seizure_diaries = np.zeros((num_theo_patients_per_trial_arm, num_baseline_scaled_time_units))
    drug_arm_testing_seizure_diaries  = np.zeros((num_theo_patients_per_trial_arm, num_testing_scaled_time_units))

    for theo_patient_index in range(num_theo_patients_per_trial_arm):

        monthly_mean    = theo_drug_arm_patient_pop_params[theo_patient_index, 0]
        monthly_std_dev = theo_drug_arm_patient_pop_params[theo_patient_index, 1]

        drug_arm_baseline_seizure_diary = \
            generate_baseline_seizure_diary(monthly_mean, 
                                            monthly_std_dev,
                                            num_baseline_months,
                                            baseline_time_scaling_const,
                                            minimum_required_baseline_seizure_count)

        drug_arm_testing_seizure_diary = \
            generate_drug_arm_testing_seizure_diary(num_testing_months, 
                                                    monthly_mean, 
                                                    monthly_std_dev, 
                                                    testing_time_scaling_const,
                                                    placebo_mu, 
                                                    placebo_sigma,
                                                    drug_mu, 
                                                    drug_sigma)
        
        drug_arm_baseline_seizure_diaries[theo_patient_index, :] = drug_arm_baseline_seizure_diary
        drug_arm_testing_seizure_diaries[theo_patient_index, :]  = drug_arm_testing_seizure_diary

    return [drug_arm_baseline_seizure_diaries, 
            drug_arm_testing_seizure_diaries  ]

