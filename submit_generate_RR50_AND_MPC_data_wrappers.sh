#!/usr/bin/bash

#SBATCH -p medium
#SBATCH --mem-per-cpu=1G
#SBATCH -t 4-00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

num_theo_patients_per_trial_arm=153
num_baseline_months=2
num_testing_months=3
minimum_required_baseline_seizure_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials_per_pop=2000
num_pops=10

data_storage_super_folder_path="/home/jmr95/rct-SNR"
block_generic_training_data_folder_name="training_data"
block_generic_testing_data_folder_name="testing_data"

num_training_compute_iters_per_loop=100
num_testing_compute_iters_per_loop=20
training_iter_slack_per_loop=10
testing_iter_slack_per_loop=4
num_blocks=75

num_actual_training_compute_iters_per_loop=$(($num_training_compute_iters_per_loop - $training_iter_slack_per_loop))
num_actual_testing_compute_iters_per_loop=$(($num_testing_compute_iters_per_loop - $testing_iter_slack_per_loop))

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_theo_patients_per_trial_arm
inputs[5]=$num_baseline_months
inputs[6]=$num_testing_months
inputs[7]=$minimum_required_baseline_seizure_count
inputs[8]=$placebo_mu
inputs[9]=$placebo_sigma
inputs[10]=$drug_mu
inputs[11]=$drug_sigma
inputs[12]=$num_trials_per_pop
inputs[13]=$num_pops
inputs[14]=$data_storage_super_folder_path
inputs[15]=$block_generic_training_data_folder_name

inputs_two[0]=$monthly_mean_min
inputs_two[1]=$monthly_mean_max
inputs_two[2]=$monthly_std_dev_min
inputs_two[3]=$monthly_std_dev_max
inputs_two[4]=$num_theo_patients_per_trial_arm
inputs_two[5]=$num_baseline_months
inputs_two[6]=$num_testing_months
inputs_two[7]=$minimum_required_baseline_seizure_count
inputs_two[8]=$placebo_mu
inputs_two[9]=$placebo_sigma
inputs_two[10]=$drug_mu
inputs_two[11]=$drug_sigma
inputs_two[12]=$num_trials_per_pop
inputs_two[13]=$num_pops
inputs_two[14]=$data_storage_super_folder_path
inputs_two[15]=$block_generic_testing_data_folder_name


for ((block_num=1; block_num<=$num_blocks; block_num=block_num+1))
do
    inputs[16]=$block_num
    inputs_two[16]=$block_num

    for ((training_compute_iter=1; training_compute_iter<$num_training_compute_iters_per_loop+1; training_compute_iter=training_compute_iter+1))
    do
        inputs[17]=$training_compute_iter

        #bash generate_RR50_and_MPC_data_wrapper.sh ${inputs[@]}
        sbatch generate_RR50_and_MPC_data_wrapper.sh ${inputs[@]}
        echo "submitted training iter #${training_compute_iter} for block #${block_num}"
    done

    for ((testing_compute_iter=1; testing_compute_iter<$num_testing_compute_iters_per_loop+1; testing_compute_iter=testing_compute_iter+1))
    do
        inputs_two[17]=$testing_compute_iter

        #bash generate_RR50_and_MPC_data_wrapper.sh ${inputs_two[@]}
        sbatch generate_RR50_and_MPC_data_wrapper.sh ${inputs_two[@]}
        echo "submitted testing iter #${testing_compute_iter} for block #${block_num}"
    done

    all_training_block_files_exist="False"
    all_testing_block_files_exist="False"
    all_block_files_exist="False"
    while [ $all_block_files_exist == "False" ]
    do
        sleep 5s
        if [ -d "${data_storage_super_folder_path}/${block_generic_training_data_folder_name}_${block_num}" ]
        then
            x1=`ls -1 "${data_storage_super_folder_path}/${block_generic_training_data_folder_name}_${block_num}/RR50_emp_stat_powers_"* | wc -l`
            x2=`ls -1 "${data_storage_super_folder_path}/${block_generic_training_data_folder_name}_${block_num}/MPC_emp_stat_powers_"* | wc -l`
            x3=`ls -1 "${data_storage_super_folder_path}/${block_generic_training_data_folder_name}_${block_num}/theo_placebo_arm_hists_"* | wc -l`
            x4=`ls -1 "${data_storage_super_folder_path}/${block_generic_training_data_folder_name}_${block_num}/theo_drug_arm_hists_"* | wc -l`
            if [ $x1 >= $num_actual_training_compute_iters_per_loop ] || [ $x2 >= $num_actual_training_compute_iters_per_loop ] || [ $x3 >= $num_actual_training_compute_iters_per_loop ] || [ $x4 >= $num_actual_training_compute_iters_per_loop ] 
            then
                all_training_block_files_exist="True"
            fi
        fi
        if [ -d "${data_storage_super_folder_path}/${block_generic_testing_data_folder_name}_${block_num}" ]
        then
            x5=`ls -1 "${data_storage_super_folder_path}/${block_generic_testing_data_folder_name}_${block_num}/RR50_emp_stat_powers_"* | wc -l`
            x6=`ls -1 "${data_storage_super_folder_path}/${block_generic_testing_data_folder_name}_${block_num}/MPC_emp_stat_powers_"* | wc -l`
            x7=`ls -1 "${data_storage_super_folder_path}/${block_generic_testing_data_folder_name}_${block_num}/theo_placebo_arm_hists_"* | wc -l`
            x8=`ls -1 "${data_storage_super_folder_path}/${block_generic_testing_data_folder_name}_${block_num}/theo_drug_arm_hists_"* | wc -l`
            if [ $x5 >= $num_actual_testing_compute_iters_per_loop ] || [ $x6 >= $num_actual_testing_compute_iters_per_loop ] || [ $x7 >= $num_actual_testing_compute_iters_per_loop ] || [ $x8 >= $num_actual_testing_compute_iters_per_loop ]
            then
                all_testing_block_files_exist="True"
            fi
        fi
        if [ $all_training_block_files_exist == "True" ] && [ $all_testing_block_files_exist == "True" ]
        then
            all_block_files_exist="True"
        else
            all_training_block_files_exist="False"
            all_testing_block_files_exist="False"
        fi
        #echo "$block_num"
        #echo "$all_training_block_files_exist"
        #echo "$all_testing_block_files_exist"
        #echo "$all_block_files_exist"
    done

done

