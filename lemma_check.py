import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import time
import psutil
import pickle as pkl


def calculate_prob_gauss(daily_mu, daily_sigma, num_patients_per_bin, num_months_per_patient, num_days_per_month, p_value_threshold):
        '''

        Inputs:

                1) daily_mu:

                        (float) - 

                2) daily_sigma:

                        (float) - 

                3) num_patients_per_bin:
                
                        (int) - 

                4) num_months_per_patient:

                        (int) - 
                
                5) num_days_per_month:

                        (int) - 

                6) p_value_threshold:

                        (float) - 

        Outputs:

                1) prob_success:

                        (float) - 

        '''
        # calculate the variance of the daily seizure counts
        daily_var = np.power(daily_sigma, 2)
                
        # check if the daily seizure count mean is not zero
        daily_mu_not_zero = daily_mu != 0

        # if it isn't, do the following:
        if( daily_mu_not_zero ):

                # check if the standard deviation is greater than the square root of the mean
                sigma_greater_than_square_root_of_mu = daily_sigma > np.sqrt(daily_mu)

                # check if the variance is not equal to the mean
                variance_not_equal_to_mu = daily_var != daily_mu

                # if both conditions, than the mean and standard deviation of the daily seizure counts can be said to be overdispersed
                fulfills_overdispersion = sigma_greater_than_square_root_of_mu and variance_not_equal_to_mu

                # if the daily seizure count data is overdispersed
                if( fulfills_overdispersion ):

                        # calculate the overdispersion parameter
                        daily_mu_squared = np.power(daily_mu, 2)
                        daily_alpha = (daily_var - daily_mu)/daily_mu_squared

                        # start keeping track of the number of times the negative binomial distribution fails to pass off as a normal distribution
                        num_failures = 0

                        # for each generated synthetic patient
                        for patient_index in range(num_patients_per_bin):
        
                                # initialize the array of monthly counts for each patient
                                monthly_counts = np.zeros(num_months_per_patient)
        
                                # for each month in the array of monthly counts
                                for month_index in range(num_months_per_patient):
                
                                        # generate a monthly count based on the gamma-poisson mixture which is equivalent to the negative binomial
                                        monthly_rate = np.random.gamma(num_days_per_month/daily_alpha, daily_alpha*daily_mu)
                                        monthly_count = np.random.poisson(monthly_rate)
                
                                        # store the monthly count
                                        monthly_counts[month_index] = monthly_count
                                
                                # the block of code commented out below was meant for kolmogorov-smirnov test, not shapiro-wilk test
                                '''
                                # calculate the estimated mean and standard deviation of the monthly counts for one patient
                                monthly_count_mean = np.mean(monthly_counts)
                                monthly_count_std = np.std(monthly_counts)

                                # if the estimated monthly count standard deviation is zero...
                                if(monthly_count_std == 0):

                                        # ... then make the estimated standard deviation into a very small positive number
                                        monthly_count_std = 0.00001
        
                                # normalize the monthly counts as if they were Gaussian data such that the normalization turns it into standard normal distributed data
                                normalized_monthly_counts = (monthly_counts - monthly_count_mean)/monthly_count_std
                                '''

                                # perform the shapiro-wilk test for comparing the normalized monthly counts of one patient to the standard normal distribution
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

        print( '\n\ndaily mu: ' + str(np.round(daily_mu*num_days_per_month, rounding_decimal_place)) + '\ndaily sigma: ' + str(np.round(daily_sigma*np.sqrt(num_days_per_month), rounding_decimal_place)) + '\nprobability of success: ' + str(np.round(100*prob_success, rounding_decimal_place)) )
        return(prob_success)


def create_prob_gauss_map(daily_mu_array, daily_sigma_array, confidence_level_percentage, num_patients_per_bin, num_months_per_patient, num_days_per_month):
        '''

        Inputs:

                1) daily_mu_array:

                        (1D Numpy array) - 
                
                2) daily_sigma_array:

                        (1D Numpy array) - 
                
                3) confidence_level_percentage:

                        (float) - 
                
                4) num_patients_per_bin:
                
                        (int) - 

                5) num_months_per_patient:

                        (int) - 
                
                6) num_days_per_month:

                        (int) - 

        Outputs:

                1) prob_successes:

                        (2D Numpy array)

        '''
        # get relevant information from arrays and initialize 2D matrix of stored probabilities
        num_daily_mus = len(daily_mu_array)
        num_daily_sigmas = len(daily_sigma_array)
        prob_successes = np.zeros((num_daily_mus, num_daily_sigmas))

        # calculate the maximum p-value needed to reject the null hypothesis that the normal and negative binomial are the same
        p_value_threshold = 1 - (confidence_level_percentage/100)

        # for each specified daily seizure count mean index
        for daily_mu_index in range(num_daily_mus):

                # for each specified daily seizure count standard deviation index
                for daily_sigma_index in range(num_daily_sigmas):

                        # get the relevant mean and standard deviation uses the respective indices
                        daily_mu = daily_mu_array[daily_mu_index]
                        daily_sigma = daily_sigma_array[daily_sigma_index]

                        prob_success = calculate_prob_gauss(daily_mu, daily_sigma, num_patients_per_bin, num_months_per_patient, num_days_per_month, p_value_threshold)

                        prob_successes[daily_mu_index, daily_sigma_index] = prob_success
        
        return prob_successes


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

                        (int) - 
                
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


def plot_data(prob_successes, monthly_mu_array, monthly_sigma_array, tick_spacing, rounding_decimal_place, model_1_monthly_count_averages, 
                model_1_monthly_count_standard_deviations, model_2_monthly_count_averages, model_2_monthly_count_standard_deviations, 
                plot_1_file_name, plot_2_file_name, prob_successes_pkl_file_name):
        '''

        Inputs:

                1) prob_successes:

                        (2D Numpy array) - 
                
                2) monthly_mu_array:

                        (1D Numpy array) - 
                
                3) monthly_sigma_array:

                        (2D Numpy array) - 
                
                4) tick_spacing:

                        (int) - 
                
                5) rounding_decimal_place:

                        (int) -
                
                6) model_1_monthly_count_averages:

                        (1D Numpy array) - 
                
                7) model_1_monthly_count_standard_deviations:

                        (1D Numpy array) - 
                
                8) model_2_monthly_count_averages:

                        (1D Numpy array) - 

                9) model_2_monthly_count_standard_deviations:

                        (1D Numpy array) - 

                10) plot_1_file_name:

                        (string) -  
                
                10) plot_2_file_name:

                        (string) -  
                
                11) prob_successes_pkl_file_name

                        (string) - 

        Outputs:

                Technically none, but practically, a graph of the lemma check map is saved in the 
                
                specified folder with the specified file name as a PNG image.


        '''

        num_monthly_mus = len(monthly_mu_array)
        num_monthly_sigmas = len(monthly_sigma_array)
        monthly_mu_ticks = np.arange(0, num_monthly_mus, tick_spacing)
        monthly_sigma_ticks = np.arange(0, num_monthly_sigmas, tick_spacing)
        monthly_mu_tick_labels = np.int_( np.round(monthly_mu_array[::tick_spacing], rounding_decimal_place ) )
        monthly_sigma_tick_labels = np.int_( np.round(monthly_sigma_array[::tick_spacing], rounding_decimal_place ) )
        monthly_mu_scatter_scaling_factor = monthly_mu_ticks[-1]/monthly_mu_array[-1]
        monthly_sigma_scatter_scaling_factor = monthly_sigma_ticks[-1]/monthly_sigma_array[-1]

        plot_1_file_path = os.getcwd() + '/' + plot_1_file_name + '.png'
        plot_2_file_path = os.getcwd() + '/' + plot_2_file_name + '.png'
        prob_successes_pkl_file_path = os.getcwd() + '/' + prob_successes_pkl_file_name + '.pkl'

        fig1 = plt.figure()

        sns.heatmap(prob_successes, cbar_kws={'label':'probability that seizure count data is Gaussian'})
        plt.scatter(model_1_monthly_count_standard_deviations*monthly_sigma_scatter_scaling_factor, model_1_monthly_count_averages*monthly_mu_scatter_scaling_factor, color='blue')
        plt.scatter(model_2_monthly_count_standard_deviations*monthly_sigma_scatter_scaling_factor, model_2_monthly_count_averages*monthly_mu_scatter_scaling_factor, color='green')
        plt.xticks( monthly_sigma_ticks, monthly_sigma_tick_labels, rotation='horizontal')
        plt.yticks( monthly_mu_ticks, monthly_mu_tick_labels )
        plt.xlabel('monthly seizure count standard deviation')
        plt.ylabel('monthly seizure count mean')
        plt.legend(['model 1 patients', 'model 2 patients'], loc='lower right')
        plt.title('Gaussianity of monthly seizure count data')
        plt.tight_layout()

        plt.savefig(plot_1_file_path)

        fig2 = plt.figure()
        sns.heatmap(prob_successes, cbar_kws={'label':'probability that seizure count data is Gaussian'})
        plt.xticks( monthly_sigma_ticks, monthly_sigma_tick_labels, rotation='horizontal')
        plt.yticks( monthly_mu_ticks, monthly_mu_tick_labels )
        plt.xlabel('daily seizure count standard deviation')
        plt.ylabel('daily seizure count mean')
        plt.title('Gaussianity of monthly seizure count data')
        plt.tight_layout()

        plt.savefig(plot_2_file_path)

        with open(prob_successes_pkl_file_path, 'wb+') as prob_successes_pkl_file:

                pkl.dump(prob_successes, prob_successes_pkl_file)


if(__name__ == '__main__'):

        start_time_in_seconds = time.time()

        # take in the command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('array', nargs='+')
        args = parser.parse_args()
        arg_array = args.array

        # exploration space for monthly seizure count mean axis of lemma check map
        monthly_mu_start = float(arg_array[0])
        monthly_mu_stop = float(arg_array[1])
        monthly_mu_step = float(arg_array[2])

        # exploration space for monthly seizure count standard deviation axis of lemma check map
        monthly_sigma_start = float(arg_array[3])
        monthly_sigma_stop = float(arg_array[4])
        monthly_sigma_step = float(arg_array[5])

        # parameters for calculating probability for each location on lemma check map
        num_patients_per_bin = int(arg_array[6])
        num_months_per_patient = int(arg_array[7])
        num_days_per_month = int(arg_array[8])
        confidence_level_percentage = float(arg_array[9])

        # number of patients to generate per model
        num_patients_per_model = int(arg_array[10])

        # plotting and presentation parameters
        rounding_decimal_place = int(arg_array[11])
        tick_spacing = int(arg_array[12])

        # file names of plots and numpy arrays to be stored
        plot_1_file_name = arg_array[13]
        plot_2_file_name = arg_array[14]
        prob_successes_pkl_file_name = arg_array[15]

        # model 1 parameters
        shape_1 = 24.143
        scale_1 = 297.366
        alpha_1 = 284.024
        beta_1 = 369.628

        # model 2 parameters
        shape_2 = 111.313
        scale_2 = 296.728
        alpha_2 = 296.339
        beta_2 = 243.719

        # initialize arrays of monthly means and standard deviations for storing axis values to be calculated over
        monthly_mu_array = np.arange(monthly_mu_start, monthly_mu_stop + monthly_mu_step, monthly_mu_step)
        monthly_sigma_array = np.arange(monthly_sigma_start, monthly_sigma_stop + monthly_sigma_step, monthly_sigma_step)

        # convert monthly seizure frequencies to daily seizure frequencies
        daily_mu_array = monthly_mu_array/num_days_per_month
        daily_sigma_array = monthly_sigma_array/np.sqrt(num_days_per_month)

        # create the map of proabilities for how probable it is that the NB data can be confused for gaussian data instead
        prob_successes = create_prob_gauss_map(daily_mu_array, daily_sigma_array, confidence_level_percentage, num_patients_per_bin, num_months_per_patient, num_days_per_month)

        # generate model 1 patient data
        [model_1_monthly_count_averages, model_1_monthly_count_standard_deviations] = generate_model_patient_data(shape_1, scale_1, alpha_1, beta_1, num_patients_per_model, num_months_per_patient, num_days_per_month)

        # generate model 2 patient data
        [model_2_monthly_count_averages, model_2_monthly_count_standard_deviations] = generate_model_patient_data(shape_2, scale_2, alpha_2, beta_2, num_patients_per_model, num_months_per_patient, num_days_per_month)

        # plot the lemma check map as well as the model 1 and model 2 patients
        plot_data(prob_successes, monthly_mu_array, monthly_sigma_array, tick_spacing, rounding_decimal_place,
                        model_1_monthly_count_averages, model_1_monthly_count_standard_deviations, 
                        model_2_monthly_count_averages, model_2_monthly_count_standard_deviations,
                        plot_1_file_name, plot_2_file_name, prob_successes_pkl_file_name)

        stop_time_in_seconds = time.time()
        total_time_in_seconds = stop_time_in_seconds - start_time_in_seconds
        total_time_in_minutes = total_time_in_seconds/60

        svem = psutil.virtual_memory()
        total_mem_in_bytes = svem.total
        available_mem_in_bytes = svem.available
        used_mem_in_bytes = total_mem_in_bytes - available_mem_in_bytes
        used_mem_in_gigabytes = used_mem_in_bytes/np.power(1024, 3)

        print('\n\ncpu time in minutes: ' + str(total_time_in_minutes) + '\nmemory usage in GB: ' + str(used_mem_in_gigabytes) + '\n' )
