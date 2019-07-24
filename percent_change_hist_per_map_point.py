import numpy as np
import matplotlib.pyplot as plt
import test_empirical_vs_analytical as teva


if(__name__=='__main__'):

    placebo_arm_monthly_mean        = 5
    placebo_arm_monthly_std_dev     = 7
    drug_arm_monthly_mean           = 6
    drug_arm_monthly_std_dev        = 7
    num_baseline_months_per_patient = 2
    num_testing_months_per_patient  = 3
    min_req_bs_sz_count             = 4
    placebo_mu                      = 0
    placebo_sigma                   = 0.05
    drug_mu                         = 0.2
    drug_sigma                      = 0.05
    num_trials                      = 5000
    num_bins                        = 200
    percentile                      = 50

    placebo_arm_monthly_mean_sq = np.power(placebo_arm_monthly_mean, 2)
    placebo_arm_monthly_var     = np.power(placebo_arm_monthly_std_dev, 2)
    drug_arm_monthly_mean_sq    = np.power(drug_arm_monthly_mean, 2)
    drug_arm_monthly_var        = np.power(drug_arm_monthly_std_dev, 2)

    placebo_arm_daily_n           = placebo_arm_monthly_mean_sq/(28*(placebo_arm_monthly_var - placebo_arm_monthly_mean))
    placebo_arm_daily_odds_ratio  = (placebo_arm_monthly_var/placebo_arm_monthly_mean) - 1
    drug_arm_daily_n              = drug_arm_monthly_mean_sq/(28*(drug_arm_monthly_var - placebo_arm_monthly_mean))
    drug_arm_daily_odds_ratio     = (drug_arm_monthly_var/drug_arm_monthly_mean) - 1
    num_baseline_days_per_patient = num_baseline_months_per_patient*28
    num_testing_days_per_patient  = num_testing_months_per_patient*28
    num_total_days_per_patient    = num_baseline_days_per_patient + num_testing_days_per_patient

    [one_placebo_map_point_percent_changes, _, 
     one_drug_map_point_percent_changes,    _] = \
         teva.generate_individual_patient_endpoints_per_map_point(placebo_arm_daily_n,
                                                                  placebo_arm_daily_odds_ratio,
                                                                  drug_arm_daily_n, 
                                                                  drug_arm_daily_odds_ratio,
                                                                  num_baseline_days_per_patient,
                                                                  num_testing_days_per_patient,
                                                                  num_total_days_per_patient,
                                                                  min_req_bs_sz_count,
                                                                  num_trials,
                                                                  placebo_mu,
                                                                  placebo_sigma,
                                                                  drug_mu,
                                                                  drug_sigma)
    
    one_placebo_map_point_percent_changes = 100*one_placebo_map_point_percent_changes
    one_drug_map_point_percent_changes    = 100*one_drug_map_point_percent_changes
    placebo_percentile = np.percentile(one_placebo_map_point_percent_changes, percentile)
    drug_percentile    = np.percentile(one_drug_map_point_percent_changes,    percentile)
    placebo_percentile_str = str(np.round(placebo_percentile, 3))
    drug_percentile_str    = str(np.round(drug_percentile,    3))

    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(one_placebo_map_point_percent_changes, bins=num_bins, density=True)
    plt.axvline(placebo_percentile, color='red')
    plt.legend(['placebo ' + str(percentile) + 'th percentile = ' + placebo_percentile_str + ' %'])
    plt.xlim([-150, 100])

    plt.subplot(2,1,2)
    plt.hist(one_drug_map_point_percent_changes, bins=num_bins, density=True)
    plt.axvline(drug_percentile, color='red')
    plt.legend(['drug ' + str(percentile) + 'th percentile = ' + drug_percentile_str + ' %'])
    plt.xlim([-150, 100])

    plt.show()

