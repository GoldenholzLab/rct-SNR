import sys
import os
import numpy as np
import json
import keras.models as models
import matplotlib.pyplot as plt


def load_iter_specific_files(data_storage_folder_name, block_num, compute_iter):

    data_storage_folder_file_path = data_storage_folder_name + '_' + str(int(block_num))

    iter_specific_theo_placebo_arm_hists_file_name = 'theo_placebo_arm_hists_' + str(compute_iter) + '.json'
    iter_specific_theo_drug_arm_hists_file_name    = 'theo_drug_arm_hists_'    + str(compute_iter) + '.json'
    iter_specific_RR50_emp_stat_powers_file_name   = 'RR50_emp_stat_powers_'   + str(compute_iter) + '.json'

    iter_specific_theo_placebo_arm_hists_file_path = data_storage_folder_file_path + '/' + iter_specific_theo_placebo_arm_hists_file_name
    iter_specific_theo_drug_arm_hists_file_path    = data_storage_folder_file_path + '/' + iter_specific_theo_drug_arm_hists_file_name
    iter_specific_RR50_emp_stat_powers_file_path   = data_storage_folder_file_path + '/' + iter_specific_RR50_emp_stat_powers_file_name

    with open(iter_specific_theo_placebo_arm_hists_file_path, 'r') as json_file:
        iter_specific_theo_placebo_arm_hists = np.array(json.load(json_file))
        
    with open(iter_specific_theo_drug_arm_hists_file_path, 'r') as json_file:
        iter_specific_theo_drug_arm_hists = np.array(json.load(json_file))
        
    with open(iter_specific_RR50_emp_stat_powers_file_path, 'r') as json_file:
        iter_specific_RR50_emp_stat_powers = np.array(json.load(json_file))
        
    return [iter_specific_theo_placebo_arm_hists,
            iter_specific_theo_drug_arm_hists,
            iter_specific_RR50_emp_stat_powers]


def collect_data_from_folder(num_monthly_means,
                             num_monthly_std_devs,
                             num_compute_iters,
                             data_storage_folder_name,
                             block_num):

    theo_placebo_arm_hists = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    theo_drug_arm_hists    = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    RR50_emp_stat_powers   = np.array([])

    for compute_iter_index in range(num_compute_iters):

        compute_iter = compute_iter_index + 1

        try:

            [iter_specific_theo_placebo_arm_hists,
             iter_specific_theo_drug_arm_hists,
             iter_specific_RR50_emp_stat_powers] = \
                 load_iter_specific_files(data_storage_folder_name, block_num, compute_iter)
        
            theo_placebo_arm_hists = np.concatenate((theo_placebo_arm_hists, iter_specific_theo_placebo_arm_hists), 2)
            theo_drug_arm_hists    = np.concatenate((theo_drug_arm_hists,    iter_specific_theo_drug_arm_hists),    2)
            RR50_emp_stat_powers   = np.concatenate((RR50_emp_stat_powers,   iter_specific_RR50_emp_stat_powers),   0)
        
        except FileNotFoundError as err:

            print('\niter #' + str(compute_iter) + ' of loop #' + str(compute_iter) + \
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
            RR50_emp_stat_powers]


def take_inputs_from_command_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    num_epochs            = int(sys.argv[5])
    num_samples_per_batch = int(sys.argv[6])

    training_data_folder_name       = sys.argv[7]
    RR50_stat_power_model_file_name = sys.argv[8]
    text_RMSEs_file_name            = sys.argv[9]

    num_compute_iters = int(sys.argv[10])
    train_block_num         = int(sys.argv[11])

    return [monthly_mean_min, monthly_mean_max, 
            monthly_std_dev_min, monthly_std_dev_max,
            num_epochs, num_samples_per_batch,
            training_data_folder_name, 
            RR50_stat_power_model_file_name, 
            text_RMSEs_file_name,
            num_compute_iters, train_block_num]


if(__name__=='__main__'):


    [monthly_mean_min, monthly_mean_max, 
     monthly_std_dev_min, monthly_std_dev_max,
     num_epochs, num_samples_per_batch,
     training_data_folder_name, 
     RR50_stat_power_model_file_name, 
     text_RMSEs_file_name,
     num_compute_iters, train_block_num] = \
         take_inputs_from_command_shell()

    text_RMSEs_file_path            = text_RMSEs_file_name + ".txt"
    num_monthly_means    = monthly_mean_max - (monthly_mean_min - 1)
    num_monthly_std_devs = monthly_std_dev_max - (monthly_std_dev_min - 1)

    [training_theo_placebo_arm_hists, 
     training_theo_drug_arm_hists, 
     training_RR50_emp_stat_powers   ] = \
         collect_data_from_folder(num_monthly_means,
                                  num_monthly_std_devs,
                                  num_compute_iters,
                                  training_data_folder_name,
                                  train_block_num)

    RR50_stat_power_model = models.load_model(RR50_stat_power_model_file_name + '.h5')
    history_object = RR50_stat_power_model.fit([training_theo_placebo_arm_hists, training_theo_drug_arm_hists], training_RR50_emp_stat_powers, epochs=num_epochs, batch_size=num_samples_per_batch)
    RR50_stat_power_model.save(RR50_stat_power_model_file_name + '.h5')

    history_dict = history_object.history
    mse_history = history_dict['loss']

    percent_rmse_history = 100*np.sqrt(mse_history)
    '''
    epoch_axis = np.arange(1, num_epochs + 1)

    print(percent_rmse_history)

    plt.figure()
    plt.plot(epoch_axis, percent_rmse_history)
    plt.savefig('rmse-per-epoch, block #' + str(loop_iter))
    '''

    if( not os.path.isfile(text_RMSEs_file_path) ):
        f = open(text_RMSEs_file_path,"w+")
        f.close()
    
    with open(text_RMSEs_file_path, 'a') as text_file:
        text_file.write('block #' + str(train_block_num) + ', RMSE: ' + str(np.round(percent_rmse_history[num_epochs-1], 3)) + ' %\n')
    