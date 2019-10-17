import sys
import os
import numpy as np
import json
import keras.models as models
import matplotlib.pyplot as plt


def load_iter_specific_files(endpoint_name, data_storage_folder_name, block_num, compute_iter):

    data_storage_folder_file_path = data_storage_folder_name + '_' + str(int(block_num))

    iter_specific_theo_placebo_arm_hists_file_name =             'theo_placebo_arm_hists_' + str(compute_iter) + '.json'
    iter_specific_theo_drug_arm_hists_file_name    =             'theo_drug_arm_hists_'    + str(compute_iter) + '.json'
    iter_specific_emp_stat_powers_file_name        = endpoint_name + '_emp_stat_powers_'   + str(compute_iter) + '.json'

    iter_specific_theo_placebo_arm_hists_file_path = data_storage_folder_file_path + '/' + iter_specific_theo_placebo_arm_hists_file_name
    iter_specific_theo_drug_arm_hists_file_path    = data_storage_folder_file_path + '/' + iter_specific_theo_drug_arm_hists_file_name
    iter_specific_emp_stat_powers_file_path        = data_storage_folder_file_path + '/' + iter_specific_emp_stat_powers_file_name

    with open(iter_specific_theo_placebo_arm_hists_file_path, 'r') as json_file:
        iter_specific_theo_placebo_arm_hists = np.array(json.load(json_file))
        
    with open(iter_specific_theo_drug_arm_hists_file_path, 'r') as json_file:
        iter_specific_theo_drug_arm_hists = np.array(json.load(json_file))
        
    with open(iter_specific_emp_stat_powers_file_path, 'r') as json_file:
        iter_specific_emp_stat_powers = np.array(json.load(json_file))
        
    return [iter_specific_theo_placebo_arm_hists,
            iter_specific_theo_drug_arm_hists,
            iter_specific_emp_stat_powers]


def collect_data_from_folder(num_monthly_means,
                             num_monthly_std_devs,
                             num_compute_iters,
                             data_storage_folder_name,
                             block_num,
                             endpoint_name):

    theo_placebo_arm_hists = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    theo_drug_arm_hists    = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    emp_stat_powers   = np.array([])

    for compute_iter_index in range(num_compute_iters):

        compute_iter = compute_iter_index + 1

        try:

            [iter_specific_theo_placebo_arm_hists,
             iter_specific_theo_drug_arm_hists,
             iter_specific_emp_stat_powers] = \
                 load_iter_specific_files(endpoint_name, data_storage_folder_name, block_num, compute_iter)
        
            theo_placebo_arm_hists = np.concatenate((theo_placebo_arm_hists, iter_specific_theo_placebo_arm_hists), 2)
            theo_drug_arm_hists    = np.concatenate((theo_drug_arm_hists,    iter_specific_theo_drug_arm_hists),    2)
            emp_stat_powers        = np.concatenate((emp_stat_powers,        iter_specific_emp_stat_powers),        0)
        
        except FileNotFoundError as err:

            print('\niter #' + str(compute_iter) + ' of block #' + str(compute_iter) + \
                  ' in either the testing or training data did not complete before the model was either tested or trained.\n' \
                  + str(err))
    
    [num_monthly_std_devs, num_monthly_means, num_samples] = theo_placebo_arm_hists.shape

    reshaped_theo_placebo_arm_hists = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))
    reshaped_theo_drug_arm_hists    = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))

    for sample_index in range(num_samples):

        reshaped_theo_placebo_arm_hists[sample_index, :, :, 0] = theo_placebo_arm_hists[:, :, sample_index]
        reshaped_theo_drug_arm_hists[sample_index, :, :, 0]    =    theo_drug_arm_hists[:, :, sample_index]

    theo_placebo_arm_hists = reshaped_theo_placebo_arm_hists
    theo_drug_arm_hists    = reshaped_theo_drug_arm_hists

    return [theo_placebo_arm_hists, 
            theo_drug_arm_hists, 
            emp_stat_powers]


def generate_model_testing_loss_and_errors(monthly_mean_min,
                                           monthly_mean_max,
                                           monthly_std_dev_min,
                                           monthly_std_dev_max,
                                           generic_stat_power_model_file_name,
                                           endpoint_name,
                                           testing_data_folder_name,
                                           num_test_compute_iters,
                                           num_test_blocks):

    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'
    num_monthly_means    = monthly_mean_max - (monthly_mean_min - 1)
    num_monthly_std_devs = monthly_std_dev_max - (monthly_std_dev_min - 1)

    stat_power_model = models.load_model(stat_power_model_file_path)

    testing_emp_stat_powers = np.array([])
    predicted_emp_stat_powers = np.array([])

    for test_block_num in range(1, num_test_blocks + 1):

        [testing_theo_placebo_arm_hists, 
         testing_theo_drug_arm_hists, 
         tmp_testing_emp_stat_powers   ] = \
             collect_data_from_folder(num_monthly_means,
                                      num_monthly_std_devs,
                                      num_test_compute_iters,
                                      testing_data_folder_name,
                                      test_block_num,
                                      endpoint_name)
        
        tmp_predicted_emp_stat_powers = np.squeeze(stat_power_model.predict([testing_theo_placebo_arm_hists, testing_theo_drug_arm_hists]))

        testing_emp_stat_powers   = np.concatenate([testing_emp_stat_powers,   tmp_testing_emp_stat_powers])
        predicted_emp_stat_powers = np.concatenate([predicted_emp_stat_powers, tmp_predicted_emp_stat_powers])

    model_errors = predicted_emp_stat_powers - testing_emp_stat_powers

    model_test_MSE = np.dot(model_errors, model_errors)/len(model_errors)
    model_test_RMSE = 100*np.sqrt(model_test_MSE)
    model_test_RMSE_str = str(np.round(model_test_RMSE, 3))

    return [model_errors, model_test_RMSE_str]
    

def take_inputs_from_command_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    testing_data_folder_name           = sys.argv[5]
    generic_stat_power_model_file_name = sys.argv[6]
    generic_text_RMSEs_file_name       = sys.argv[7]
    endpoint_name                      = sys.argv[8]

    num_test_compute_iters = int(sys.argv[9])
    num_test_blocks        = int(sys.argv[10])

    num_bins = int(sys.argv[10])

    return [monthly_mean_min,    monthly_mean_max,
            monthly_std_dev_min, monthly_std_dev_max,
            testing_data_folder_name, 
            generic_stat_power_model_file_name, 
            generic_text_RMSEs_file_name,
            endpoint_name, num_test_compute_iters, 
            num_test_blocks, num_bins]


if(__name__=='__main__'):

    [monthly_mean_min,    monthly_mean_max,
     monthly_std_dev_min, monthly_std_dev_max,
     testing_data_folder_name, 
     generic_stat_power_model_file_name, 
     generic_text_RMSEs_file_name,
     endpoint_name, num_test_compute_iters, 
     num_test_blocks, num_bins] = \
         take_inputs_from_command_shell()

    [model_errors, model_test_RMSE_str] = \
        generate_model_testing_loss(monthly_mean_min,
                              monthly_mean_max,
                              monthly_std_dev_min,
                              monthly_std_dev_max,
                              generic_stat_power_model_file_name,
                              endpoint_name,
                              testing_data_folder_name,
                              num_test_compute_iters,
                              num_test_blocks)

    text_RMSEs_file_path = endpoint_name + '_' + generic_text_RMSEs_file_name + ".txt"
    with open(text_RMSEs_file_path, 'a') as text_file:
         text_file.write('testing RMSE: ' + model_test_RMSE_str + ' %')

    # the plan is to eventually out the plotting code into a separate script....
    model_percent_errors = 100*model_errors
    plt.figure()
    plt.hist(model_percent_errors, bins=num_bins, density=True)
    plt.xlabel('percent error')
    plt.title('histogram of ' + endpoint_name + ' statistical power prediction error')
    plt.savefig(endpoint_name + ' statistical power prediction error histogram')
