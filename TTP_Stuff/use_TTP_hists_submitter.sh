#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load python/3.6.0
module load conda2/4.2.13
module load R/3.5.1
source activate main_env

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

num_baseline_months_per_patient=2
num_testing_months_per_patient=3
min_req_base_sz_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=10000
num_theo_patients_per_trial_arm=153
num_patients_per_dot=5000
alpha=0.05

#hist_maps_folder="$(pwd)/hist_maps_folder"
#stat_power_storage_folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'
hist_maps_folder="$/n/scratch2/jmr95/rct-SNR-9_3_2019/hist_maps_folder"
stat_power_storage_folder='/n/scratch2/jmr95/rct-SNR-9_3_2019/TTP_stat_power_estimates'
num_stat_power_estimates=1000

inputs[0]=$monthly_mean_min
inputs[1]=$monthly_mean_max
inputs[2]=$monthly_std_dev_min
inputs[3]=$monthly_std_dev_max
inputs[4]=$num_baseline_months_per_patient
inputs[5]=$num_testing_months_per_patient
inputs[6]=$min_req_base_sz_count
inputs[7]=$placebo_mu
inputs[8]=$placebo_sigma
inputs[9]=$drug_mu
inputs[10]=$drug_sigma
inputs[11]=$num_trials
inputs[12]=$num_theo_patients_per_trial_arm
inputs[13]=$num_patients_per_dot
inputs[14]=$alpha
inputs[15]=$hist_maps_folder
inputs[16]=$stat_power_storage_folder

for ((stat_power_estimate_index=1; stat_power_estimate_index<=$num_stat_power_estimates ; stat_power_estimate_index=stat_power_estimate_index+1))
do
    inputs[17]=$stat_power_estimate_index

    #python -u use_TTP_hists.py ${inputs[@]}
    sbatch use_TTP_hists_wrapper.sh ${inputs[@]}

done