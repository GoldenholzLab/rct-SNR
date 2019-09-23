import numpy as np
from seizure_diary_generation import generate_placebo_arm_seizure_diary
from seizure_diary_generation import generate_drug_arm_seizure_diary


def calculate_percent_change(baseline_seizure_diary,
                             testing_seizure_diary):

    baseline_seizure_frequency = np.mean(baseline_seizure_diary)
    testing_seizure_frequency = np.mean(testing_seizure_diary)

    if(baseline_seizure_frequency == 0):
        baseline_seizure_frequency = 0.00000001

    percent_change = np.divide(baseline_seizure_frequency - testing_seizure_frequency, baseline_seizure_frequency)

    return percent_change


if(__name__=='__main__'):

    monthly_mean = 28
    monthly_std_dev = 5.3
    num_baseline_months = 2
    num_testing_months = 3
    baseline_time_scaling_const = 28
    testing_time_scaling_const = 28
    minimum_required_baseline_seizure_count = 4
    placebo_mu = 0
    placebo_sigma = 0.05
    drug_mu = 0.2
    drug_sigma = 0.05

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
    
    placebo_arm_percent_change = \
        calculate_percent_change(placebo_arm_baseline_seizure_diary,
                                 placebo_arm_testing_seizure_diary)
    
    drug_arm_percent_change = \
        calculate_percent_change(drug_arm_baseline_seizure_diary,
                                 drug_arm_testing_seizure_diary)

    print(100*np.array([placebo_arm_percent_change, drug_arm_percent_change]))
