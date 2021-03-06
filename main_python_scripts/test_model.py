import sys
import os
import numpy as np
import json
import keras.models as models
import matplotlib.pyplot as plt
from data_collector import collect_keras_formatted_samples_and_labels

# the nrxt two lines were put in to make sure this code works on mac laptops
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
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
                             num_compute_iters_per_block,
                             data_storage_folder_name,
                             block_num,
                             endpoint_name):

    theo_placebo_arm_hists = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    theo_drug_arm_hists    = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    emp_stat_powers   = np.array([])

    for compute_iter_index in range(num_compute_iters_per_block):

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

    keras_formatted_theo_placebo_arm_hists = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_theo_drug_arm_hists    = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))

    for sample_index in range(num_samples):

        keras_formatted_theo_placebo_arm_hists[sample_index, :, :, 0] = theo_placebo_arm_hists[:, :, sample_index]
        keras_formatted_theo_drug_arm_hists[sample_index, :, :, 0]    =    theo_drug_arm_hists[:, :, sample_index]

    return [keras_formatted_theo_placebo_arm_hists, 
            keras_formatted_theo_drug_arm_hists,
            emp_stat_powers]
'''

def generate_model_testing_loss_and_errors(monthly_mean_lower_bound,
                                           monthly_mean_upper_bound,
                                           monthly_std_dev_lower_bound,
                                           monthly_std_dev_upper_bound,
                                           generic_stat_power_model_file_name,
                                           endpoint_name,
                                           data_storage_folder_name,
                                           num_train_compute_iters_per_block,
                                           num_test_compute_iters_per_block,
                                           leftout_testing_block_nums):

    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'
    num_monthly_means    = monthly_mean_upper_bound    - (monthly_mean_lower_bound    - 1)
    num_monthly_std_devs = monthly_std_dev_upper_bound - (monthly_std_dev_lower_bound - 1)

    stat_power_model = models.load_model(stat_power_model_file_path)

    testing_emp_stat_powers   = np.array([])
    predicted_emp_stat_powers = np.array([])

    for block_num in leftout_testing_block_nums:

        '''
        [keras_formatted_testing_theo_placebo_arm_hists, 
         keras_formatted_testing_theo_drug_arm_hists,
         tmp_testing_emp_stat_powers] = \
             collect_data_from_folder(num_monthly_means,
                                      num_monthly_std_devs,
                                      num_test_compute_iters_per_block,
                                      testing_data_folder_name,
                                      block_num,
                                      endpoint_name)
        '''

        [keras_formatted_testing_theo_placebo_arm_hists, 
         keras_formatted_testing_theo_drug_arm_hists,
         tmp_testing_emp_stat_powers]  = \
             collect_keras_formatted_samples_and_labels(data_storage_folder_name,
                                                        endpoint_name,
                                                        block_num,
                                                        num_train_compute_iters_per_block,
                                                        num_test_compute_iters_per_block,
                                                        num_monthly_means,
                                                        num_monthly_std_devs)
        
        tmp_predicted_emp_stat_powers = np.squeeze(stat_power_model.predict([keras_formatted_testing_theo_placebo_arm_hists, keras_formatted_testing_theo_drug_arm_hists]))

        testing_emp_stat_powers   = np.concatenate([testing_emp_stat_powers,   tmp_testing_emp_stat_powers])
        predicted_emp_stat_powers = np.concatenate([predicted_emp_stat_powers, tmp_predicted_emp_stat_powers])

    model_errors = predicted_emp_stat_powers - testing_emp_stat_powers

    model_test_MSE = np.dot(model_errors, model_errors)/len(model_errors)
    model_test_RMSE = 100*np.sqrt(model_test_MSE)
    model_test_RMSE_str = str(np.round(model_test_RMSE, 3))

    return [model_errors, model_test_RMSE_str]
    

def take_inputs_from_command_shell():

    monthly_mean_lower_bound    = int(sys.argv[1])
    monthly_mean_upper_bound    = int(sys.argv[2])
    monthly_std_dev_lower_bound = int(sys.argv[3])
    monthly_std_dev_upper_bound = int(sys.argv[4])

    data_storage_folder_name           = sys.argv[5]
    generic_stat_power_model_file_name = sys.argv[6]
    generic_text_RMSEs_file_name       = sys.argv[7]
    model_errors_file_name             = sys.argv[8]

    num_train_compute_iters_per_block = int(sys.argv[9])
    num_test_compute_iters_per_block  = int(sys.argv[10])
    num_total_blocks                  = int(sys.argv[11])
    endpoint_name                     =     sys.argv[12]

    random_training_block_nums = list(map(int, list(filter( lambda a: a != '', sys.argv[13].split('\'') ))))#sys.argv[12]

    leftout_testing_block_nums = []
    for block_index in range(1, num_total_blocks + 1):
        if(random_training_block_nums.count(block_index) == 0):
            leftout_testing_block_nums.append(block_index)
    
    print(random_training_block_nums)
    print(leftout_testing_block_nums)

    return [monthly_mean_lower_bound,    
            monthly_mean_upper_bound,
            monthly_std_dev_lower_bound, 
            monthly_std_dev_upper_bound,
            data_storage_folder_name, 
            generic_stat_power_model_file_name, 
            generic_text_RMSEs_file_name,
            model_errors_file_name,
            num_train_compute_iters_per_block,
            num_test_compute_iters_per_block, 
            endpoint_name,
            leftout_testing_block_nums]


if(__name__=='__main__'):

    [monthly_mean_lower_bound,    
     monthly_mean_upper_bound,
     monthly_std_dev_lower_bound, 
     monthly_std_dev_upper_bound,
     data_storage_folder_name, 
     generic_stat_power_model_file_name, 
     generic_text_RMSEs_file_name,
     model_errors_file_name,
     num_train_compute_iters_per_block,
     num_test_compute_iters_per_block, 
     endpoint_name,
     leftout_testing_block_nums] = \
         take_inputs_from_command_shell()

    [model_errors, model_test_RMSE_str] = \
         generate_model_testing_loss_and_errors(monthly_mean_lower_bound,
                                                monthly_mean_upper_bound,
                                                monthly_std_dev_lower_bound,
                                                monthly_std_dev_upper_bound,
                                                generic_stat_power_model_file_name,
                                                endpoint_name,
                                                data_storage_folder_name,
                                                num_train_compute_iters_per_block,
                                                num_test_compute_iters_per_block,
                                                leftout_testing_block_nums)

    text_RMSEs_file_path = endpoint_name + '_' + generic_text_RMSEs_file_name + ".txt"
    with open(text_RMSEs_file_path, 'a') as text_file:
         text_file.write('testing RMSE: ' + model_test_RMSE_str + ' %')

    with open(endpoint_name + '_' + model_errors_file_name + '.json', 'w+') as json_file:
        json.dump(model_errors.tolist(), json_file)
