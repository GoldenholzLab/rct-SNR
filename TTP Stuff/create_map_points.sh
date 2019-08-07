

monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

monthly_mean=6
monthly_std_dev=6

min_req_base_sz_count=4
num_baseline_months_per_patient=2
num_testing_months_per_patient=3
num_theo_patients_per_trial_arm=153

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05

num_trials=2
folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'

inputs[3]=$min_req_base_sz_count
inputs[4]=$num_baseline_months_per_patient
inputs[5]=$num_testing_months_per_patient
inputs[6]=$num_theo_patients_per_trial_arm
inputs[7]=$placebo_mu
inputs[8]=$placebo_sigma
inputs[9]=$drug_mu
inputs[10]=$drug_sigma
inputs[11]=$num_trials
inputs[12]=$folder

for ((monthly_mean=$monthly_mean_min; monthly_mean<$monthly_mean_max+1; monthly_mean=$monthly_mean+1));
do
    for ((monthly_std_dev=$monthly_std_dev_min; monthly_std_dev<$monthly_std_dev_max+1; monthly_std_dev=monthly_std_dev+1));
    do
        sqrt_monthly_mean=$(bc <<< "scale=6;sqrt($monthly_mean)")
        if (( $(echo "$monthly_std_dev>$sqrt_monthly_mean" | bc -l) ));
        then

            inputs[1]=$monthly_mean
            inputs[2]=$monthly_std_dev

            bash create_map_point_wrapper.sh ${inputs[@]}
        else

            bash create_null_map_point_wrapper.sh $monthly_mean $monthly_std_dev $folder
        fi
    done
done

