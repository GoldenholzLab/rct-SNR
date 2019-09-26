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
module load conda2/4.2.13
module load python/3.6.0
source activate main_env
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

num_trials=500
num_patients_per_map_location=50
alpha=0.05

folder=/n/scratch2/jmr95/rct-SNR-data/TTP-data/TTP-mapsls
RMSE_folder=/n/scratch2/jmr95/rct-SNR-data/TTP-data/TTP-power-estims
num_iter=10

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
inputs[12]=$num_trials
inputs[13]=$num_patients_per_map_location
inputs[14]=$alpha
inputs[15]=$folder
inputs[16]=$RMSE_folder

for ((estim_iter=1; estim_iter<=num_iter; estim_iter=estim_iter+1))
do
    inputs[17]=$estim_iter

    sbatch TTP_emp_and_map_power_estimation_wrapper.sh ${inputs[@]}
done

