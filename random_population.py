import numpy as np
import matplotlib.pyplot as plt

if(__name__ == '__main__'):

    monthly_mean_min = 1
    monthly_mean_max = 16
    monthly_std_dev_min = 1
    monthly_std_dev_max = 16
    num_patients_per_trial_arm = 153

    patient_pop_params = np.zeros((num_patients_per_trial_arm, 2))

    for patient_index in range(num_patients_per_trial_arm):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        patient_pop_params[patient_index, 0] = monthly_mean
        patient_pop_params[patient_index, 1] = monthly_std_dev
    
    plt.figure()
    plt.scatter(patient_pop_params[:, 0], patient_pop_params[:, 1])
    plt.show()
