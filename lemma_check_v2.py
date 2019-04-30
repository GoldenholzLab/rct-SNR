import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_prob_gauss(monthly_mu, monthly_sigma, num_patients_per_bin, num_months_per_patient, p_value_threshold, rounding_decimal_place):
        '''

        Inputs:

                1) monthly_mu:

                        (float) - 

                2) monthly_sigma:

                        (float) - 

                3) num_patients_per_bin:
                
                        (int) - 

                4) num_months_per_patient:

                        (int) - 
                
                5) p_value_threshold:

                        (float) - 

        Outputs:

                1) prob_success:

                        (float) - 

        '''
        # calculate the variance of the daily seizure counts
        monthly_var = np.power(monthly_sigma, 2)
                
        # check if the daily seizure count mean is not zero
        monthly_mu_not_zero = monthly_mu != 0

        # if it isn't, do the following:
        if( monthly_mu_not_zero ):

                # check if the standard deviation is greater than the square root of the mean, which is equivalent to overdispersion
                sigma_greater_than_square_root_of_mu = monthly_sigma > np.sqrt(monthly_mu)

                # if the daily seizure count data is overdispersed
                if( sigma_greater_than_square_root_of_mu ):

                        # calculate the overdispersion parameter
                        monthly_mu_squared = np.power(monthly_mu, 2)
                        monthly_alpha = (monthly_var - monthly_mu)/monthly_mu_squared

                        # start keeping track of the number of times the negative binomial distribution fails to pass off as a normal distribution
                        num_failures = 0

                        # for each generated synthetic patient
                        for patient_index in range(num_patients_per_bin):
        
                                # initialize the array of monthly counts for each patient
                                monthly_counts = np.zeros(num_months_per_patient)
        
                                # for each month in the array of monthly counts
                                for month_index in range(num_months_per_patient):
                
                                        # generate a monthly count based on the gamma-poisson mixture which is equivalent to the negative binomial
                                        monthly_rate = np.random.gamma(1/monthly_alpha, monthly_alpha*monthly_mu)
                                        monthly_count = np.random.poisson(monthly_rate)
                
                                        # store the monthly count
                                        monthly_counts[month_index] = monthly_count

                                # perform the shapiro-wilk test for comparing the monthly counts of one patient to the standard normal distribution
                                [_, p_value] = stats.shapiro(monthly_counts)
        
                                # check if the p-value indicates whether or not the null hypothesis of the seizure count data was rejected or not
                                rejected_normality = p_value < p_value_threshold
        
                                # if the null hypothesis of the Gaussianity of the seizure data was rejected
                                if(rejected_normality):
                
                                        # increase the number of failures by one
                                        num_failures = num_failures + 1

                        # convert the number of failures to the number of successes
                        num_successes = num_patients_per_bin - num_failures

                        # calculate the probability of the monthly count data successdully passing off as Gaussian-distributed
                        prob_success = num_successes/num_patients_per_bin

                else:

                        # if the overdisperion conditions are not fulfilled, then calculating the probability does not make sense
                        prob_success = np.NaN
        
        else:

                # if the true mean is zero, then calculating the probability does not make sense either
                prob_success = np.NaN

        print( '\n\ndaily mu: '             + str( np.round( monthly_mu,        rounding_decimal_place) ) + 
               '\ndaily sigma: '            + str( np.round( monthly_sigma,     rounding_decimal_place) ) + 
               '\nprobability of success: ' + str( np.round( 100*prob_success,  rounding_decimal_place) )   )

        return(prob_success)


def generate_prob_gauss_map(monthly_mu_axis_start,    monthly_mu_axis_stop,      monthly_mu_axis_step, 
                            monthly_sigma_axis_start, monthly_sigma_axis_stop,   monthly_sigma_axis_step,
                            num_patients_per_bin, num_months_per_patient, confidence_level_interval, rounding_decimal_place):

    # calculate the p-value threshold based on the specified confidence level
    p_value_threshold = 1 - ( confidence_level_interval/100 )

    # create the monthely mean and monthly standard devation axes
    monthly_mu_array = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_axis_step, monthly_mu_axis_step)
    monthly_sigma_array = np.flip( np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_axis_step, monthly_sigma_axis_step), 0 )

    # get the lengths of these axes
    num_monthly_mu = len(monthly_mu_array)
    num_monthly_sigma = len(monthly_sigma_array)

    # initialize the 2D numpy array in which the porbabilities will be stored
    data_matrix = np.zeros((num_monthly_sigma, num_monthly_mu))

    # over each monthly mean index:
    for monthly_sigma_index in range(num_monthly_sigma):

            # over each monthly standard deviation index:
            for monthly_mu_index in range(num_monthly_mu):
                
                    # extract the current monthly mean and monthly standard deviation using the relevant indices
                    monthly_mu = monthly_mu_array[monthly_mu_index]
                    monthly_sigma = monthly_sigma_array[monthly_sigma_index]

                    # actually calculate the probability and store it in the 2D numpy array
                    data_matrix[monthly_sigma_index, monthly_mu_index] = \
                        calculate_prob_gauss(monthly_mu, monthly_sigma, num_patients_per_bin, num_months_per_patient, p_value_threshold, rounding_decimal_place)
    
    return [num_monthly_mu, num_monthly_sigma, data_matrix]


def generate_model_monthly_patient_data(shape, scale, alpha, beta, num_patients_per_model, num_months_per_patient):
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

                        (int) - 
                
                6) num_months_per_patient:

                        (int) - 

        Outputs:

                1) model_monthly_count_averages:

                        (1D Numpy array) - 
                
                2) model_standard_deviations:

                        (1D Numpy array) - 

        '''
        
        # specify the number of days in a month
        num_days_per_month = 28

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


def generate_prob_gauss_plot(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, 
                             num_patients_per_model, plot_models,
                             monthly_mu_axis_start,    monthly_mu_axis_stop,    monthly_mu_axis_step, 
                             monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step,
                             num_patients_per_bin, num_months_per_patient, confidence_level_interval, rounding_decimal_place):

    # generate the plot of probabilities as to whether data is gaussian or not
    [num_monthly_mu, num_monthly_sigma, data_matrix] = generate_prob_gauss_map(monthly_mu_axis_start,    monthly_mu_axis_stop,      monthly_mu_axis_step, 
                                                                                monthly_sigma_axis_start, monthly_sigma_axis_stop,   monthly_sigma_axis_step,
                                                                                num_patients_per_bin, num_months_per_patient, confidence_level_interval, rounding_decimal_place)

    # generate the tick labels for the monthly mean axis as well as for the monthly standard deviation axis
    # the tick labels are distinctly different from the actual ticks
    monthly_mu_tick_labels = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_tick_spacing, monthly_mu_tick_spacing)
    monthly_sigma_tick_labels = np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_tick_spacing, monthly_sigma_tick_spacing)

    # get the number of tick labels for both axes
    num_monthly_mu_tick_labels = len(monthly_mu_tick_labels)
    num_monthly_sigma_tick_labels = len(monthly_sigma_tick_labels)

    # calculate the ticks (location of each tick) for the monthly mean axis and monthly standard deviation axis
    monthly_mu_ticks = np.append( np.array([0]), monthly_mu_tick_labels[1:]/monthly_mu_axis_step + np.ones(num_monthly_mu_tick_labels - 1) )
    monthly_sigma_ticks = np.flip( np.append( np.array([0]), monthly_sigma_tick_labels[1:]/monthly_sigma_axis_step + np.ones(num_monthly_sigma_tick_labels - 1) ), 0)

    # plot the heatmap using seaborn and the relevant ticks\tick-labels
    plt.figure()
    ax = sns.heatmap(data_matrix)
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')

    if(plot_models):

        # generate model 1 monthly patient averages and monthly standard deviations
        [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = \
            generate_model_monthly_patient_data(shape_1, scale_1, alpha_1, beta_1, num_patients_per_model, num_months_per_patient)
    
        # generate model 2 monthly patient averages and monthly standard deviations
        [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = \
            generate_model_monthly_patient_data(shape_2, scale_2, alpha_2, beta_2, num_patients_per_model, num_months_per_patient)

        # get the scaling ratios for any other data that needs to be plotted on top of the heatmap
        monthly_mu_scale_ratio = num_monthly_mu/(monthly_mu_axis_stop - monthly_mu_axis_start)
        monthly_sigma_scale_ratio = num_monthly_sigma/(monthly_sigma_axis_stop - monthly_sigma_axis_start)

        # plot the monthl averages and monthly standard devations of model 1 patients
        plt.scatter(model_1_monthly_count_averages*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - model_1_monthly_count_standard_deviations)*monthly_sigma_scale_ratio )

        # plot the monthl averages and monthly standard devations of model 2 patients
        plt.scatter(model_2_monthly_count_averages*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - model_2_monthly_count_standard_deviations)*monthly_sigma_scale_ratio )

    plt.show()


# heatmap parameters
monthly_mu_axis_start = 0
monthly_mu_axis_stop = 16
monthly_mu_axis_step = 1

monthly_sigma_axis_start = 0
monthly_sigma_axis_stop = 16
monthly_sigma_axis_step = 1

monthly_mu_tick_spacing = 2
monthly_sigma_tick_spacing = 2

num_patients_per_bin = 1000
num_months_per_patient = 24
confidence_level_interval = 95
rounding_decimal_place = 2

# model 1 and model 2 parameters
shape_1 = 24.143
scale_1 = 297.366
alpha_1 = 284.024
beta_1 = 369.628
shape_2 = 111.313
scale_2 = 296.728
alpha_2 = 296.339
beta_2 = 243.719
num_patients_per_model = 1000
plot_models = True

generate_prob_gauss_plot(shape_1, scale_1, alpha_1, beta_1, shape_2, scale_2, alpha_2, beta_2, 
                         num_patients_per_model, plot_models,
                         monthly_mu_axis_start,    monthly_mu_axis_stop,    monthly_mu_axis_step, 
                         monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step,
                         num_patients_per_bin, num_months_per_patient, confidence_level_interval, rounding_decimal_place)