import json
import numpy as np

def collect_samples_and_labels(data_storage_folder_name,
                               endpoint_name,
                               training_or_testing,
                               block_num,
                               compute_iter):

    theo_placebo_arm_hists_file_name =                 'theo_placebo_arm_hists_' + str(compute_iter) + '.json'
    theo_drug_arm_hists_file_name    =                 'theo_drug_arm_hists_'    + str(compute_iter) + '.json'
    emp_stat_powers_file_name        = endpoint_name + '_emp_stat_powers_'       + str(compute_iter) + '.json'

    theo_placebo_arm_hists_file_path = data_storage_folder_name + str(block_num) + '/' + theo_placebo_arm_hists_file_name
    theo_drug_arm_hists_file_path    = data_storage_folder_name + str(block_num) + '/' + theo_drug_arm_hists_file_name
    emp_stat_powers_file_path        = data_storage_folder_name + str(block_num) + '/' + emp_stat_powers_file_name

    with open(theo_placebo_arm_hists_file_path, 'r') as json_file:
        theo_placebo_arm_hists = np.array(json.load(json_file))
        
    with open(theo_drug_arm_hists_file_path, 'r') as json_file:
        theo_drug_arm_hists = np.array(json.load(json_file))
        
    with open(emp_stat_powers_file_path, 'r') as json_file:
        emp_stat_powers = np.array(json.load(json_file))

    return [theo_placebo_arm_hists, 
            theo_drug_arm_hists, 
            emp_stat_powers]


def collect_samples_and_labels_from_folder(data_storage_folder_name,
                                           endpoint_name,
                                           training_or_testing,
                                           block_num,
                                           num_iters,
                                           num_monthly_means,
                                           num_monthly_std_devs):

    theo_placebo_arm_hists = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    theo_drug_arm_hists    = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    emp_stat_powers        = np.array([])

    for compute_iter in np.arange(1, num_iters + 1, 1):

        try:

            [tmp_theo_placebo_arm_hists, 
             tmp_theo_drug_arm_hists, 
             tmp_emp_stat_powers] = \
                 collect_samples_and_labels(data_storage_folder_name,
                                            endpoint_name,
                                            training_or_testing,
                                            block_num,
                                            compute_iter)
    
            theo_placebo_arm_hists = np.concatenate((theo_placebo_arm_hists, tmp_theo_placebo_arm_hists), 2)
            theo_drug_arm_hists    = np.concatenate((theo_drug_arm_hists,    tmp_theo_drug_arm_hists),    2)
            emp_stat_powers        = np.concatenate((emp_stat_powers,        tmp_emp_stat_powers),        0)
        
        except FileNotFoundError:

            print('Missing data in block #'  + str(block_num))
    
    return [theo_placebo_arm_hists,
            theo_drug_arm_hists,
            emp_stat_powers]


def collect_samples_and_labels_from_block(data_storage_folder_name,
                                          endpoint_name,
                                          block_num,
                                          num_training_iters,
                                          num_testing_iters,
                                          num_monthly_means,
                                          num_monthly_std_devs):

    [training_theo_placebo_arm_hists,
     training_theo_drug_arm_hists,
     training_emp_stat_powers] = \
         collect_samples_and_labels_from_folder(data_storage_folder_name  + '/training_data_',
                                                endpoint_name,
                                                'training',
                                                block_num,
                                                num_training_iters,
                                                num_monthly_means,
                                                num_monthly_std_devs)
    
    [testing_theo_placebo_arm_hists,
     testing_theo_drug_arm_hists,
     testing_emp_stat_powers] = \
         collect_samples_and_labels_from_folder(data_storage_folder_name + '/testing_data_',
                                                endpoint_name,
                                                'testing',
                                                block_num,
                                                num_testing_iters,
                                                num_monthly_means,
                                                num_monthly_std_devs)
    
    theo_placebo_arm_hists = np.concatenate((training_theo_placebo_arm_hists, testing_theo_placebo_arm_hists), 2)
    theo_drug_arm_hists    = np.concatenate((training_theo_drug_arm_hists,    testing_theo_drug_arm_hists),    2)
    emp_stat_powers        = np.concatenate((training_emp_stat_powers, testing_emp_stat_powers),               0)

    return [theo_placebo_arm_hists, 
            theo_drug_arm_hists, 
            emp_stat_powers]


def collect_keras_formatted_samples_and_labels(data_storage_folder_name,
                                               endpoint_name,
                                               block_num,
                                               num_training_iters,
                                               num_testing_iters,
                                               num_monthly_means,
                                               num_monthly_std_devs):

    [theo_placebo_arm_hists, 
     theo_drug_arm_hists, 
     emp_stat_powers] = \
         collect_samples_and_labels_from_block(data_storage_folder_name,
                                               endpoint_name,
                                               block_num,
                                               num_training_iters,
                                               num_testing_iters,
                                               num_monthly_means,
                                               num_monthly_std_devs)
    
    num_samples = len(emp_stat_powers)

    keras_formatted_theo_placebo_arm_hists = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))
    keras_formatted_theo_drug_arm_hists    = np.zeros((num_samples, num_monthly_std_devs, num_monthly_means, 1))

    #import pandas as pd

    for sample_index in range(num_samples):

        '''
        print('\n' + pd.DataFrame(theo_placebo_arm_hists[:, :, sample_index]).to_string() + '\n\n' + \
                     pd.DataFrame(theo_drug_arm_hists[:, :, sample_index]).to_string()    + '\n\n' + \
                     str(100*emp_stat_powers[sample_index])                                   + '\n'     )
        '''
        keras_formatted_theo_placebo_arm_hists[sample_index, :, :, 0] = theo_placebo_arm_hists[:, :, sample_index]
        keras_formatted_theo_drug_arm_hists[sample_index, :, :, 0]    =    theo_drug_arm_hists[:, :, sample_index]

    return [keras_formatted_theo_placebo_arm_hists,
            keras_formatted_theo_drug_arm_hists,
            emp_stat_powers]


if(__name__=='__main__'):

    data_storage_folder_name = '/Users/juanromero/Documents/rct-SNR_O2_generated_data/keras_data_and_labels_11-14-2019'

    endpoint_name = 'RR50'

    import sys
    block_num = int(sys.argv[1])

    num_training_iters = 15
    num_testing_iters = 5
    num_monthly_means = 16
    num_monthly_std_devs = 16

    [keras_formatted_theo_placebo_arm_hists,
     keras_formatted_theo_drug_arm_hists,
     emp_stat_powers] = \
         collect_keras_formatted_samples_and_labels(data_storage_folder_name,
                                                    endpoint_name,
                                                    block_num,
                                                    num_training_iters,
                                                    num_testing_iters,
                                                    num_monthly_means,
                                                    num_monthly_std_devs)

