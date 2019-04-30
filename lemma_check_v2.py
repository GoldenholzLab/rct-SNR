import numpy as np
import scipy.stats as stats
import pandas as pd


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

    p_value_threshold = 1 - ( confidence_level_interval/100 )

    monthly_mu_array = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_axis_step, monthly_mu_axis_step)
    monthly_sigma_array = np.flip( np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_axis_step, monthly_sigma_axis_step), 0 )

    num_monthly_mu = len(monthly_mu_array)
    num_monthly_sigma = len(monthly_sigma_array)

    data_matrix = np.zeros((num_monthly_sigma, num_monthly_mu))

    for monthly_sigma_index in range(num_monthly_sigma):

            for monthly_mu_index in range(num_monthly_mu):
                
                    monthly_mu = monthly_mu_array[monthly_mu_index]
                    monthly_sigma = monthly_sigma_array[monthly_sigma_index]

                    data_matrix[monthly_sigma_index, monthly_mu_index] = \
                        calculate_prob_gauss(monthly_mu, monthly_sigma, num_patients_per_bin, num_months_per_patient, p_value_threshold, rounding_decimal_place)
    
    return data_matrix


# heatmap parameters
monthly_mu_axis_start = 0
monthly_mu_axis_stop = 16
monthly_mu_axis_step = 1

monthly_sigma_axis_start = 0
monthly_sigma_axis_stop = 16
monthly_sigma_axis_step = 1

num_patients_per_bin = 1000
num_months_per_patient = 24
confidence_level_interval = 95
rounding_decimal_place = 2

data_matrix = generate_prob_gauss_map(monthly_mu_axis_start,    monthly_mu_axis_stop,      monthly_mu_axis_step, 
                                      monthly_sigma_axis_start, monthly_sigma_axis_stop,   monthly_sigma_axis_step,
                                      num_patients_per_bin, num_months_per_patient, confidence_level_interval, rounding_decimal_place)

print(pd.DataFrame(data_matrix).to_string())
