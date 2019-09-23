import numpy as np


def generate_seizure_diary(num_months, 
                           monthly_mean, 
                           monthly_std_dev, 
                           time_scaling_const):

    num_scaled_time_units = num_months*time_scaling_const
    seizure_diary = np.zeros(num_scaled_time_units)

    monthly_var = np.power(monthly_std_dev, 2)
    monthly_mean_sq  = np.power(monthly_mean, 2)
    monthly_overdispersion = (monthly_var - monthly_mean)/monthly_mean_sq
    monthly_n = 1/monthly_overdispersion
    odds_ratio = monthly_overdispersion*monthly_mean

    for scaled_time_unit_index in range(num_scaled_time_units):

        time_scaled_rate = np.random.gamma(monthly_n/time_scaling_const, odds_ratio)
        time_scaled_count = np.random.poisson(time_scaled_rate)

        seizure_diary[scaled_time_unit_index] = time_scaled_count
    
    return seizure_diary


def apply_effect(seizure_diary,
                 num_months,
                 time_scaling_const,
                 effect):

    num_scaled_time_units = num_months*time_scaling_const

    for scaled_time_unit_index in range(num_scaled_time_units):

        num_removed = 0

        for seizure_index in range(np.int_(seizure_diary[scaled_time_unit_index])):

            prob = np.random.uniform(0, 1)

            if(prob < np.abs(effect)):

                num_removed = num_removed + np.sign(effect)
        
        seizure_diary[scaled_time_unit_index] = seizure_diary[scaled_time_unit_index] - num_removed

    return seizure_diary


def generate_seizure_diary_with_minimum_count(num_months,
                                              monthly_mean,
                                              monthly_std_dev,
                                              time_scaling_const,
                                              minimum_required_seizure_count):

    acceptable_count = False

    while(not acceptable_count):

        baseline_seizure_diary = \
            generate_seizure_diary(num_months, 
                                   monthly_mean, 
                                   monthly_std_dev, 
                                   time_scaling_const)
    
        num_seizures = np.sum(baseline_seizure_diary)

        if(num_seizures >= minimum_required_seizure_count):

            acceptable_count = True
    
    return baseline_seizure_diary


def generate_placebo_arm_seizure_diary(monthly_mean, 
                                       monthly_std_dev,
                                       num_baseline_months,
                                       num_testing_months,
                                       baseline_time_scaling_const,
                                       testing_time_scaling_const,
                                       minimum_required_baseline_seizure_count,
                                       placebo_mu,
                                       placebo_sigma):

    placebo_effect = np.random.normal(placebo_mu, placebo_sigma)

    placebo_arm_baseline_seizure_diary = \
        generate_seizure_diary_with_minimum_count(num_baseline_months, 
                                                  monthly_mean, 
                                                  monthly_std_dev, 
                                                  baseline_time_scaling_const,
                                                  minimum_required_baseline_seizure_count)
    
    placebo_arm_baseline_seizure_diary = \
        apply_effect(placebo_arm_baseline_seizure_diary,
                     num_baseline_months,
                     baseline_time_scaling_const,
                     placebo_effect)

    placebo_arm_testing_seizure_diary = \
        generate_seizure_diary(num_testing_months, 
                               monthly_mean, 
                               monthly_std_dev, 
                               testing_time_scaling_const)

    placebo_arm_testing_seizure_diary = \
        apply_effect(placebo_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     placebo_effect)
    
    return [placebo_arm_baseline_seizure_diary, placebo_arm_testing_seizure_diary]


def generate_drug_arm_seizure_diary(monthly_mean, 
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

    placebo_effect = np.random.normal(placebo_mu, placebo_sigma)
    drug_effect = np.random.normal(drug_mu, drug_sigma)

    drug_arm_baseline_seizure_diary = \
        generate_seizure_diary_with_minimum_count(num_baseline_months, 
                                                  monthly_mean, 
                                                  monthly_std_dev, 
                                                  baseline_time_scaling_const,
                                                  minimum_required_baseline_seizure_count)

    drug_arm_baseline_seizure_diary = \
        apply_effect(drug_arm_baseline_seizure_diary,
                     num_baseline_months,
                     baseline_time_scaling_const,
                     placebo_effect)
    
    drug_arm_testing_seizure_diary = \
        generate_seizure_diary(num_testing_months, 
                               monthly_mean, 
                               monthly_std_dev, 
                               testing_time_scaling_const)
    
    drug_arm_testing_seizure_diary = \
        apply_effect(drug_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     placebo_effect)
    
    drug_arm_testing_seizure_diary = \
        apply_effect(drug_arm_testing_seizure_diary,
                     num_testing_months,
                     testing_time_scaling_const,
                     drug_effect)
    
    return [drug_arm_baseline_seizure_diary, drug_arm_testing_seizure_diary]

