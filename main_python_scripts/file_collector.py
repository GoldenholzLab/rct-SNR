import numpy as np
import json


def load_iter_specific_files(data_storage_folder_name, loop_iter, compute_iter):

    data_storage_folder_file_path = './' + data_storage_folder_name + '_' + str(int(loop_iter))

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
                             loop_iter):

    theo_placebo_arm_hists = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    theo_drug_arm_hists    = np.array(num_monthly_std_devs*[num_monthly_means*[num_monthly_means*[]]])
    RR50_emp_stat_powers   = np.array([])

    for compute_iter_index in range(num_compute_iters):

        compute_iter = compute_iter_index + 1

        [iter_specific_theo_placebo_arm_hists,
         iter_specific_theo_drug_arm_hists,
         iter_specific_RR50_emp_stat_powers] = \
             load_iter_specific_files(data_storage_folder_name, loop_iter, compute_iter)
        
        theo_placebo_arm_hists = np.concatenate((theo_placebo_arm_hists, iter_specific_theo_placebo_arm_hists), 2)
        theo_drug_arm_hists    = np.concatenate((theo_drug_arm_hists,    iter_specific_theo_drug_arm_hists),    2)
        RR50_emp_stat_powers   = np.concatenate((RR50_emp_stat_powers,   iter_specific_RR50_emp_stat_powers),   0)
    
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
    