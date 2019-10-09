import sys
import os
sys.path.insert(0, os.getcwd())
from main_python_scripts.file_collector import load_iter_specific_files
from main_python_scripts.file_collector import collect_data_from_folder
import keras.models as models
#import shutil


def take_inputs_from_command_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    training_data_folder_name = sys.argv[5]
    RR50_stat_power_model_file_name = sys.argv[6]
    num_compute_iters = int(sys.argv[7])
    loop_iter         = int(sys.argv[8])

    return [monthly_mean_min,    monthly_mean_max,
            monthly_std_dev_min, monthly_std_dev_max,
            training_data_folder_name, num_compute_iters,
            RR50_stat_power_model_file_name, loop_iter]


if(__name__=='__main__'):

    [monthly_mean_min,    monthly_mean_max,
     monthly_std_dev_min, monthly_std_dev_max,
     training_data_folder_name, num_compute_iters,
     RR50_stat_power_model_file_name, loop_iter]= \
          take_inputs_from_command_shell()

    num_monthly_means    = monthly_mean_max - (monthly_mean_min - 1)
    num_monthly_std_devs = monthly_std_dev_max - (monthly_std_dev_min - 1)

    [theo_placebo_arm_hists, 
     theo_drug_arm_hists, 
     RR50_emp_stat_powers   ] = \
         collect_data_from_folder(num_monthly_means,
                                  num_monthly_std_devs,
                                  num_compute_iters,
                                  training_data_folder_name,
                                  loop_iter)
    
    RR50_stat_power_model = models.load_model(RR50_stat_power_model_file_name + '_' + str(int(loop_iter)) + '.h5')
    RR50_stat_power_model.fit([theo_placebo_arm_hists, theo_drug_arm_hists], RR50_emp_stat_powers, epochs=3, batch_size=5)
    RR50_stat_power_model.save(RR50_stat_power_model_file_name + '_' + str(int(loop_iter + 1)) + '.h5')
    
    #shutil.rmtree(training_data_folder_name + '_' + str(int(loop_iter)))
