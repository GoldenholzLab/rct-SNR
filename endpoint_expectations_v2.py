import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin,
                                num_months_per_patient_baseline, num_months_per_patient_testing):
    '''

    Inputs:

        1) monthly_mu:

            (float) - 

        2) monthly_sigma:

            (float) - 
        
        3) num_patients_per_trial:

            (int) - 
        
        4) num_trials_per_bin:

            (int) - 
        
        5) num_months_per_patient_baseline:

            (int) - 
        
        6) num_months_per_patient_testing:

            (int) - 

    Outputs:

        1) expected_RR50:

            (float) - 
        
        2) expected_MPC:

            (float) - 

    '''

    # calculate the total number of months
    num_months_per_patient_total = num_months_per_patient_baseline + num_months_per_patient_testing

    # check to see if the mean is not zero
    monthly_mu_not_zero = monthly_mu != 0

    # if the mean is not zero...
    if( monthly_mu_not_zero ):

        # check to see if the daily standard deviation is greater than the square root of the daily mean
        monthly_sigma_greater_than_square_root_of_monthly_mu = monthly_sigma > np.sqrt(monthly_mu)

        # if the daily standard deviation is greater than the square root of the daily mean...
        if( monthly_sigma_greater_than_square_root_of_monthly_mu ):

            # calculate the overdispersion parameter
            monthly_mu_squared = np.power(monthly_mu, 2)
            monthly_var = np.power(monthly_sigma, 2)
            monthly_alpha = (monthly_var - monthly_mu)/monthly_mu_squared

            # initialize the arrays that will store the RR50 and MPC from each trial
            RR50_array = np.zeros(num_trials_per_bin)
            MPC_array = np.zeros(num_trials_per_bin)

            # for every trial...
            for trial_index in range(num_trials_per_bin):

                # initialize the 2D array of monthly counts from multiple patients over one trial
                monthly_counts = np.zeros((num_patients_per_trial, num_months_per_patient_total))

                # for each individual patient
                for patient_index in range(num_patients_per_trial):

                    # for each month in each individual patient's diary
                    for month_index in range(num_months_per_patient_total):

                        # generate a monthly count according to gamma-poisson mixture
                        monthly_rate = np.random.gamma(1/monthly_alpha, monthly_alpha*monthly_mu)
                        monthly_count = np.random.poisson(monthly_rate)

                        # store the monthly count
                        monthly_counts[patient_index, month_index] = monthly_count 

                # separate the monthly counts into baseline and testing periods
                baseline_monthly_counts = monthly_counts[:, 0:num_months_per_patient_baseline]
                testing_monthly_counts = monthly_counts[:, num_months_per_patient_baseline:]
        
                # calculate the seizure frequencies of the basline and test period for each patient
                baseline_monthly_frequencies = np.mean(baseline_monthly_counts, 1)
                testing_monthly_frequencies = np.mean(testing_monthly_counts, 1)

                # for each patient
                for patient_index in range(num_patients_per_trial):

                    # if the baseline seizure frequency turns out to zero...
                    if( baseline_monthly_frequencies[patient_index] == 0 ):

                        # make the baseline frequency into a very small positive number
                        baseline_monthly_frequencies[patient_index] = 0.000000000000000001
        
                # calculate the percent changes of the seizure frequencies across the baseline and testing periods for each patient
                percent_changes = np.divide(baseline_monthly_frequencies - testing_monthly_frequencies, baseline_monthly_frequencies)

                # calculate the RR50 and MPC for this trial
                RR50 = 100*np.sum(percent_changes >= 0.5)/num_patients_per_trial
                MPC = 100*np.median(percent_changes)

                # store the RR50 and MPC
                RR50_array[trial_index] = RR50
                MPC_array[trial_index] = MPC
        
            # calculate the mean RR50 and mean MPC over all the trials
            expected_RR50 = np.mean(RR50_array)
            expected_MPC = np.mean(MPC_array)

            return [expected_RR50, expected_MPC]
        
        else:

            # if the square root of the daily mean is less than the daily standard deviation, then the data is not underdispersed
            # therefore, it does not make sense to model this mean-and-standard-deviation pair
            return [np.NaN, np.NaN]
    
    else:

        # if the daily mean is zero, then it also does not make sense to model that pair
        return [np.NaN, np.NaN]


def generate_expected_endpoint_maps(monthly_mu_axis_start, monthly_mu_axis_stop, monthly_mu_axis_step, 
                                    monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step, 
                                    num_patients_per_trial, num_trials_per_bin,
                                    num_months_per_patient_baseline, num_months_per_patient_testing):

    '''

    Inputs:

        1) monthly_mu_axis_start:

            (float) - 

        2) monthly_mu_axis_stop: 
        
            (float) - 
        
        3) monthly_mu_axis_step:

            (float) - 

        4)  monthly_sigma_axis_start:
        
            (float) - 
        
        5) monthly_sigma_axis_stop:
        
            (float) - 
        
        6) monthly_sigma_axis_step:

            (float) - 

        7) num_patients_per_trial:
        
            (float) - 
        
        8) num_trials_per_bin:
        
            (float) -  

        9) num_months_per_patient_baseline:
        
            (float) - 
        
        10) num_months_per_patient_testing:

            (float) - 

    Outputs:
    
        1) num_monthly_mu:
        
            (int) - 

        2) num_monthly_sigma:

            (int) -  
        
        3)  monthly_mu_array:

            (1D Numpy array) - 

        3) RR50_matrix:

            (2D Numpy array) - 
        
        4) MPC_matrix:

            (2D Numpy array) - 

    '''

    # create the monthly mean and monthly standard deviation axes
    monthly_mu_array = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_axis_step, monthly_mu_axis_step)
    monthly_sigma_array = np.flip( np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_axis_step, monthly_sigma_axis_step), 0 )

    # get the lengths of these axes
    num_monthly_mu = len(monthly_mu_array)
    num_monthly_sigma = len(monthly_sigma_array)

    # initialize the two 2D numpy array in which the expected RR50s and the expected MPCs will be stored
    expected_RR50_map = np.zeros((num_monthly_sigma, num_monthly_mu))
    expected_MPC_map = np.zeros((num_monthly_sigma, num_monthly_mu))

    # over each monthly mean index:
    for monthly_sigma_index in range(num_monthly_sigma):

        # over each monthly standard deviation index:
        for monthly_mu_index in range(num_monthly_mu):

            # extract the current monthly mean and monthly standard deviation using the relevant indices
            monthly_mu = monthly_mu_array[monthly_mu_index]
            monthly_sigma = monthly_sigma_array[monthly_sigma_index]

            # actually calculate the expected RR50 as well as the expected MPC
            [expected_RR50, expected_MPC] = estimate_expected_endpoints(monthly_mu, monthly_sigma, num_patients_per_trial, num_trials_per_bin,
                                                                        num_months_per_patient_baseline, num_months_per_patient_testing)
            
            print('\n\nmonthly mean: '             + str(monthly_mu)    + 
                    '\nmonthly standard deviation: ' + str(monthly_sigma) + 
                    '\nexpected RR50: '            + str(expected_RR50) + 
                    '\nexpected MPC: '             + str(expected_MPC)    )
            
            # store the expected RR50 and expected MPC into their respective 2D arrays
            expected_RR50_map[monthly_sigma_index, monthly_mu_index] = expected_RR50
            expected_MPC_map[monthly_sigma_index, monthly_mu_index] = expected_MPC
    
    return [num_monthly_mu, num_monthly_sigma, monthly_mu_array, expected_RR50_map, expected_MPC_map]


def generate_model_patient_data(shape, scale, alpha, beta, num_patients_per_model, num_months_per_patient, num_days_per_month):
    '''

    Inputs:

        1) shape:

            (float) -
                
        2) scale:

            (float) - 
                
        3) alpha:

            (float) - 
                
        4) beta:

            (float) - 

        5) num_patients:

            int) - 
                
        6) num_months_per_patient:

            (int) - 
                
        7) num_days_per_month

            (int) - 

    Outputs:

        1) model_monthly_count_averages:

            (1D Numpy array) - 
                
        2) model_standard_deviations:

            (1D Numpy array) - 

    '''
        
    # initialize array of monthly counts
    monthly_counts = np.zeros((num_patients_per_model, num_months_per_patient))
        
    # over each patient
    for patient_index in range(num_patients_per_model):

        # generate an n and p parameter for each patient
        n = np.random.gamma(shape, 1/scale)
        p = np.random.beta(alpha, beta)

        # for each month in each patient's diary
        for month_index in range(num_months_per_patient):

            # generate a monthly count as according to n and parameters and gamma-poisson mixture model
            monthly_rate = np.random.gamma(num_days_per_month*n, (1 - p)/p)
            monthly_count = np.random.poisson(monthly_rate)

            # store monthly_count
            monthly_counts[patient_index, month_index] = monthly_count
        
    # calculate the average and standard deviation of the monthly seizure counts for each patient
    model_monthly_count_averages = np.mean(monthly_counts, 1)
    model_monthly_count_standard_deviations = np.std(monthly_counts, 1)

    return [model_monthly_count_averages, model_monthly_count_standard_deviations]


def generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, 
                        num_patients_per_model, num_months_per_patient, num_days_per_month,
                        monthly_mu_axis_start, monthly_mu_axis_stop, monthly_mu_axis_step, 
                        monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step,
                        num_patients_per_trial, num_trials_per_bin,
                        num_months_per_patient_baseline, num_months_per_patient_testing):

    '''

    Inputs:

        1) shape_1:
        
            (float) - 
        
        2) scale_1:
        
            (float) - 
        
        3) alpha_1:
        
            (float) - 
        
        4) beta_1:
        
            (float) - 

        5) shape_2:
        
            (float) - 
        
        6) scale_2:
        
            (float) - 

        7) alpha_2:
        
            (float) - 
        
        8) beta_2:

            (float) -  
        
        9) num_patients_per_model:
        
            (int) - 
        
        10) num_months_per_patient:
        
            (int) - 
        
        11) num_days_per_month:

            (int) - 

        12) monthly_mu_axis_start:
        
            (float) - 
            
        13) monthly_mu_axis_stop:
        
            (float) - 
            
        14) monthly_mu_axis_step:

            (float) - 

        15) monthly_sigma_axis_start:
        
            (float) - 
        
        16) monthly_sigma_axis_stop:
        
            (float) - 
        
        17) monthly_sigma_axis_step:

            (float) - 

        18) num_patients_per_trial:
        
            (int) -  
            
        19) num_trials_per_bin:

            (int) - 
        
        20) num_months_per_patient_baseline:
        
            (int) - 
        
        21) num_months_per_patient_testing:

            (int) - 

    Outputs:
    
        1) num_monthly_mu:
        
            (int) - 

        2) num_monthly_sigma:

            (int) - 
        
        3) monthly_mu_array:

            (1D Numpy array) - 

        4) Model_1_expected_RR50:
            
            (float) - 
        
        5) Model_1_expected_MPC:
        
            (float) - 
        
        6) Model_2_expected_RR50
        
            (float) - 

        7) Model_2_expected_MPC:
        
            (float) - 
        
        8) expected_RR50_map:
        
            (2D Numpy array) - 
        
        9) expected_MPC_map:
        
            (2D Numpy array) - 
        
        10) H_model_1:
            
            (2D Numpy array) - 
        
        11) H_model_2:

            (2D Numpy array) - 

    '''

    # generate the expected RR50 and expected MPC maps
    [num_monthly_mu, num_monthly_sigma, monthly_mu_array, expected_RR50_map, expected_MPC_map] = \
        generate_expected_endpoint_maps(monthly_mu_axis_start, monthly_mu_axis_stop, monthly_mu_axis_step, 
                                        monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step, 
                                        num_patients_per_trial, num_trials_per_bin,
                                        num_months_per_patient_baseline, num_months_per_patient_testing)

    # generate Model 1 patients
    [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = \
                                        generate_model_patient_data(shape_1, scale_1, alpha_1, beta_1,
                                                                    num_patients_per_model, num_months_per_patient, num_days_per_month)

    # generate Model 2 patients
    [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = \
                                    generate_model_patient_data(shape_2, scale_2, alpha_2, beta_2,
                                                                num_patients_per_model, num_months_per_patient, num_days_per_month)

    # get the size of those expected endpoint maps
    [nx, ny] = expected_RR50_map.shape

    # calculate the histogram of the Model 1 patients
    [H_model_1, _, _] = np.histogram2d(model_1_monthly_count_averages, model_1_monthly_count_standard_deviations, bins=[ny, nx], 
                                        range=[[monthly_mu_axis_start, monthly_mu_axis_stop], [monthly_sigma_axis_start, monthly_sigma_axis_stop]])
    H_model_1 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_1))))
    norm_const_1 = np.sum(np.sum(H_model_1, 0))
    H_model_1 = H_model_1/norm_const_1

    # calculate the histogram of the Model 2 patients
    [H_model_2, _, _] = np.histogram2d(model_2_monthly_count_averages, model_2_monthly_count_standard_deviations, bins=[ny, nx], 
                                    range=[[monthly_mu_axis_start, monthly_mu_axis_stop], [monthly_sigma_axis_start, monthly_sigma_axis_stop]])
    H_model_2 = np.flipud(np.fliplr(np.transpose(np.flipud(H_model_2))))
    norm_const_2 = np.sum(np.sum(H_model_2, 0))
    H_model_2 = H_model_2/norm_const_2

    # combine the expected endpoint maps as well as the Model 1 and Model 2 histograms to get the expected RR50 and MPC for both models
    Model_1_expected_RR50 = np.sum(np.nansum(np.multiply(H_model_1, expected_RR50_map), 0))
    Model_1_expected_MPC  = np.sum(np.nansum(np.multiply(H_model_1, expected_MPC_map),  0))
    Model_2_expected_RR50 = np.sum(np.nansum(np.multiply(H_model_2, expected_RR50_map), 0))
    Model_2_expected_MPC  = np.sum(np.nansum(np.multiply(H_model_2, expected_MPC_map),  0))

    return [num_monthly_mu, num_monthly_sigma, monthly_mu_array, Model_1_expected_RR50, 
            Model_1_expected_MPC, Model_2_expected_RR50, Model_2_expected_MPC, 
            expected_RR50_map, expected_MPC_map, H_model_1, H_model_2]


num_patients_per_trial = 153
num_trials_per_bin = 100
num_months_per_patient_baseline = 2
num_months_per_patient_testing = 3

# heatmap parameters
monthly_mu_axis_start = 0
monthly_mu_axis_stop = 16
monthly_mu_axis_step = 1

monthly_sigma_axis_start = 0
monthly_sigma_axis_stop = 16
monthly_sigma_axis_step = 1

monthly_mu_tick_spacing = 1
monthly_sigma_tick_spacing = 1

# model 1 and model 2 parameters
shape_1 = 24.143
scale_1 = 297.366
alpha_1 = 284.024
beta_1 = 369.628

shape_2 = 111.313
scale_2 = 296.728
alpha_2 = 296.339
beta_2 = 243.719

num_patients_per_model = 10000
num_months_per_patient = 24
num_days_per_month = 28

# power law parameters
max_power_law_slope = 1.3
min_power_law_slope = 0.5
power_law_slope_spacing = 0.1
legend_decimal_round = 1

# picture file names
expected_RR50_filename = 'expected_RR50'
expected_MPC_filename = 'expected_MPC'
expected_RR50_with_curves_filename = 'expected_RR50_with_power_law_curves'
expected_MPC_with_curves_filename = 'expected_MPC_with_power_law_curves'
model_1_2D_hist_filename = 'Model_1_2D_histogram'
model_2_2D_hist_filename = 'Model_2_2D_histogram'

# calculate the expected enpoint response maps, the Model 1 and Model 2 histograms, and the 
[num_monthly_mu, num_monthly_sigma, monthly_mu_array, 
 Model_1_expected_RR50, Model_1_expected_MPC, Model_2_expected_RR50, Model_2_expected_MPC, 
 expected_RR50_map, expected_MPC_map, H_model_1, H_model_2] = \
    generate_SNR_data(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, 
                        num_patients_per_model, num_months_per_patient, num_days_per_month,
                        monthly_mu_axis_start, monthly_mu_axis_stop, monthly_mu_axis_step, 
                        monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step,
                        num_patients_per_trial, num_trials_per_bin,
                        num_months_per_patient_baseline, num_months_per_patient_testing)

# generate the tick labels for the monthly mean axis as well as for the monthly standard deviation axis
# the tick labels are distinctly different from the actual ticks
monthly_mu_tick_labels = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_tick_spacing, monthly_mu_tick_spacing)
monthly_sigma_tick_labels = np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_tick_spacing, monthly_sigma_tick_spacing)

# get the number of tick labels for both axes
num_monthly_mu_tick_labels = len(monthly_mu_tick_labels)
num_monthly_sigma_tick_labels = len(monthly_sigma_tick_labels)

# calculate the ticks (location of each tick) for the monthly mean axis and monthly standard deviation axis
monthly_mu_ticks = monthly_mu_tick_labels/monthly_mu_axis_step + 0.5*np.ones(num_monthly_mu_tick_labels)
monthly_sigma_ticks = np.flip( monthly_sigma_tick_labels/monthly_sigma_axis_step + 0.5*np.ones(num_monthly_sigma_tick_labels), 0)

# plot the expected RR50
plt.figure()
ax = sns.heatmap(expected_RR50_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Expected RR50 Placebo')
plt.savefig(os.getcwd + '/' + expected_RR50_filename + '.png')

# plot the expected MPC
plt.figure()
ax = sns.heatmap(expected_MPC_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Expected MPC Placebo')
plt.savefig(os.getcwd + '/' + expected_MPC_filename + '.png')

# get the scaling ratios for any other data that needs to be plotted on top of the heatmap
monthly_mu_scale_ratio = num_monthly_mu/(monthly_mu_axis_stop - monthly_mu_axis_start)
monthly_sigma_scale_ratio = num_monthly_sigma/(monthly_sigma_axis_stop - monthly_sigma_axis_start)
power_law_slopes = np.arange(min_power_law_slope/power_law_slope_spacing, max_power_law_slope/power_law_slope_spacing + 1, 1)*power_law_slope_spacing

# plot the expected RR50 with corresponding power law curves
plt.figure()
ax = sns.heatmap(expected_RR50_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
for power_law_slope_index in range(len(power_law_slopes)):
        power_law_slope = power_law_slopes[power_law_slope_index]
        power_law_monthly_sigma_array = np.power(monthly_mu_array, power_law_slope)
        plt.plot(monthly_mu_array*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - power_law_monthly_sigma_array)*monthly_sigma_scale_ratio )
plt.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] )
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Expected RR50 Placebo')
plt.savefig(os.getcwd + '/' + expected_RR50_with_curves_filename + '.png')

# plot the expected MPC with corresponding power law curves
plt.figure()
ax = sns.heatmap(expected_MPC_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
for power_law_slope_index in range(len(power_law_slopes)):
        power_law_slope = power_law_slopes[power_law_slope_index]
        power_law_monthly_sigma_array = np.power(monthly_mu_array, power_law_slope)
        plt.plot(monthly_mu_array*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - power_law_monthly_sigma_array)*monthly_sigma_scale_ratio )
plt.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] ) 
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Expected MPC Placebo')
plt.savefig(os.getcwd + '/' + expected_MPC_with_curves_filename + '.png')

# plot the 2D histogram of Model 1 patients
plt.figure()
ax = sns.heatmap(H_model_1, cbar_kws={'label':'probability of sampling patient from a point'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Model 1 patient population')
plt.savefig(os.getcwd + '/' + model_1_2D_hist_filename + '.png')

# plot the 2D histogram of Model 2 patients
plt.figure()
ax = sns.heatmap(H_model_2, cbar_kws={'label':'probability of sampling patient from a point'})
ax.set_xticks(monthly_mu_ticks)
ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
ax.set_yticks(monthly_sigma_ticks)
ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
plt.xlabel('monthly seizure count mean')
plt.ylabel('monthly seizure count standard deviation')
plt.title('Model 2 patient population')
plt.savefig(os.getcwd + '/' + model_2_2D_hist_filename + '.png')


