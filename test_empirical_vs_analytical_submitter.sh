#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:03
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

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_theo_patients_per_trial_arm=153
num_baseline_months_per_patient=2
num_testing_months_per_patient=3
min_req_bs_sz_count=4

num_trials=5000
num_iter=50

inputs[1]=$monthly_mean_min
inputs[2]=$monthly_mean_max
inputs[3]=$monthly_std_dev_min
inputs[4]=$monthly_std_dev_max
inputs[5]=$placebo_mu
inputs[6]=$placebo_sigma
inputs[7]=$drug_mu
inputs[8]=$drug_sigma
inputs[9]=$num_theo_patients_per_trial_arm
inputs[10]=$num_baseline_months_per_patient
inputs[11]=$num_testing_months_per_patient
inputs[12]=$min_req_bs_sz_count
inputs[13]=$num_trials

for ((i=1; i<num_iter+1; i=i+1))
do
    inputs[14]=$i
    sbatch test_empirical_vs_analytical_wrapper.sh ${inputs[@]}

done