import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


num_base_months = 2
num_test_months = 3
model_1_mean = 2.7
model_2_mean = 8.5

num_patients_per_bin = 5000
num_patients_model_1 = 5000
num_patients_model_2 = 5000
hist_bins = 50

mu_sz_start = 0
mu_sz_stop = 18
mu_sz_step = 0.5

sigma_sz_start = 0
sigma_sz_stop = 10
sigma_sz_step = 0.5

num_monthly_counts = num_base_months + num_test_months

mu_sz_begin = mu_sz_start + mu_sz_step
mu_sz_end = mu_sz_stop + mu_sz_step
mu_endpoint = False
num_mu_sz = np.int_( (mu_sz_end - mu_sz_begin)/mu_sz_step )

sigma_sz_begin = sigma_sz_start + sigma_sz_step
sigma_sz_end = sigma_sz_stop + sigma_sz_step
sigma_endpoint = False
num_sigma_sz = np.int_( (sigma_sz_end - sigma_sz_begin)/sigma_sz_step )

mu_sz_array = np.linspace(mu_sz_begin, mu_sz_end, num_mu_sz, mu_endpoint)
sigma_sz_array = np.flip( np.linspace(sigma_sz_begin, sigma_sz_end, num_sigma_sz, sigma_endpoint), 0 )

median_pc_matrix = np.zeros([num_sigma_sz, num_mu_sz])
sigma_pc_matrix = np.zeros([num_sigma_sz, num_mu_sz])
snr_matrix = np.zeros([num_sigma_sz, num_mu_sz])

for sigma_sz_index in range(num_sigma_sz):

    for mu_sz_index in range(num_mu_sz):

        mu_sz = mu_sz_array[mu_sz_index]
        sigma_sz = sigma_sz_array[sigma_sz_index]

        percent_change_array = np.zeros(num_patients_per_bin)

        for patient_index in range(num_patients_per_bin):

            monthly_counts = np.zeros(num_monthly_counts)

            for monthly_count_index in range(num_monthly_counts):

                acceptable_count = False

                while(not acceptable_count):

                    monthly_count = np.random.normal(mu_sz, sigma_sz)

                    if( monthly_count >=0 ):
                        
                        acceptable_count = True
                
                monthly_counts[monthly_count_index] = monthly_count

            base_monthly_counts = monthly_counts[0:num_base_months]
            test_monthly_counts = monthly_counts[num_base_months:]

            base_monthly_freq = np.mean(base_monthly_counts)
            test_monthly_freq = np.mean(test_monthly_counts)

            percent_change = 100*(base_monthly_freq  - test_monthly_freq)/base_monthly_freq

            percent_change_array[patient_index] = percent_change

        median_pc = np.median(percent_change_array)
        sigma_pc = np.std(percent_change_array)

        rel_std_dev_pc = np.abs(median_pc)/sigma_pc

        median_pc_matrix[sigma_sz_index, mu_sz_index] = median_pc
        sigma_pc_matrix[sigma_sz_index, mu_sz_index] = sigma_pc
        snr_matrix[sigma_sz_index, mu_sz_index] = rel_std_dev_pc


model_1_monthly_patient_counts = np.zeros([num_patients_model_1, num_monthly_counts])
for patient in range(num_patients_model_1):

    for monthly_count_index in range(num_monthly_counts):

        acceptable_count = False

        while(not acceptable_count):

            monthly_count = np.random.normal(model_1_mean, model_1_mean**0.7)

            if(monthly_count >= 0):

                acceptable_count = True

        model_1_monthly_patient_counts[patient, monthly_count_index] = monthly_count

model_1_patient_count_averages = np.mean(model_1_monthly_patient_counts, 1)
model_1_patient_count_std_devs = np.std(model_1_monthly_patient_counts, 1)

[H_model_1, x_edges_1, y_edges_1] = np.histogram2d(model_1_patient_count_averages, model_1_patient_count_std_devs, [num_sigma_sz, num_mu_sz], [[mu_sz_start, mu_sz_stop], [sigma_sz_start, sigma_sz_stop]], True)
H_model_1 = np.flipud(H_model_1)

norm_const_1 = np.sum(np.sum(H_model_1, 0))
bin_area_1 = (x_edges_1[1] - x_edges_1[0])*(y_edges_1[1] - y_edges_1[0])
H_model_1 = H_model_1*bin_area_1
average_snr_model_1 = np.mean(np.mean(np.multiply(H_model_1, snr_matrix), 0))


model_2_monthly_patient_counts = np.zeros([num_patients_model_2, num_monthly_counts])
for patient in range(num_patients_model_2):

    for monthly_count_index in range(num_monthly_counts):

        acceptable_count = False

        while(not acceptable_count):

            monthly_count = np.random.normal(model_2_mean, model_2_mean**0.7)

            if(monthly_count >= 0):

                acceptable_count = True

        model_2_monthly_patient_counts[patient, monthly_count_index] = monthly_count

model_2_patient_count_averages = np.mean(model_2_monthly_patient_counts, 1)
model_2_patient_count_std_devs = np.std(model_2_monthly_patient_counts, 1)

[H_model_2, x_edges_2, y_edges_2] = np.histogram2d(model_2_patient_count_averages, model_2_patient_count_std_devs, [num_sigma_sz, num_mu_sz], [[mu_sz_start, mu_sz_stop], [sigma_sz_start, sigma_sz_stop]], True)
H_model_2 = np.flipud(H_model_2)
bin_area_2 = (x_edges_2[1] - x_edges_2[0])*(y_edges_2[1] - y_edges_2[0])
H_model_2 = H_model_2*bin_area_2
average_snr_model_2 = np.mean(np.mean(np.multiply(H_model_2, snr_matrix), 0))


fig1 = plt.figure()
plt.imshow(median_pc_matrix, cmap='coolwarm')
plt.xticks( np.arange(1, 19, 1)/mu_sz_step - 1, np.arange(1, 21, 1))
plt.xlabel('seizure count mean')
plt.yticks(np.arange(0, 10, 1)/sigma_sz_step, np.flip(np.arange(1, 11, 1), 0))
plt.ylabel('seizure count standard deviation')
plt.colorbar()
plt.title('median of percent change')

fig2 = plt.figure()
plt.imshow(sigma_pc_matrix, cmap='coolwarm')
plt.xticks( np.arange(1, 19, 1)/mu_sz_step - 1, np.arange(1, 21, 1))
plt.xlabel('seizure count mean')
plt.yticks(np.arange(0, 10, 1)/sigma_sz_step, np.flip(np.arange(1, 11, 1), 0))
plt.ylabel('seizure count standard deviation')
plt.colorbar()
plt.title('standard deviation of percent change')

fig3 = plt.figure()
plt.imshow(snr_matrix, cmap='coolwarm')
plt.xticks( np.arange(1, 19, 1)/mu_sz_step - 1, np.arange(1, 21, 1))
plt.xlabel('seizure count mean')
plt.yticks(np.arange(0, 10, 1)/sigma_sz_step, np.flip(np.arange(1, 11, 1), 0))
plt.ylabel('seizure count standard deviation')
plt.colorbar()
plt.title('SNR of percent change')

#print('\n\n' + str([average_snr_model_1, average_snr_model_2]) + '\n\n')
print('\n\nsnr for model 1: ' + str(average_snr_model_1) + '\nsnr for model 2: ' + str(average_snr_model_2) + '\n\n')

'''
plt.figure()
plt.hist2d(model_1_patient_count_averages, model_1_patient_count_std_devs, [num_x_bins, num_y_bins], range = [[mu_sz_start, mu_sz_stop], [sigma_sz_start, sigma_sz_stop]], normed='True')
plt.xlabel('monthly count averages')
plt.ylabel('monthly count standard deviations')
plt.title('heatmap of model 1 patients')

plt.figure()
plt.hist2d(model_2_patient_count_averages, model_2_patient_count_std_devs, [2*hist_bins, hist_bins], range = [[mu_sz_start, mu_sz_stop], [sigma_sz_start, sigma_sz_stop]], normed='True')
plt.xlabel('monthly count averages')
plt.ylabel('monthly count standard deviations')
plt.title('heatmap of model 2 patients')
'''

plt.show()


