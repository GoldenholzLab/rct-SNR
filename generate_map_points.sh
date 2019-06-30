
start_monthly_mean=0
stop_monthly_mean=16
step_monthly_mean=1

start_monthly_std_dev=0
stop_monthly_std_dev=16
step_monthly_std_dev=1

num_trials=10

directory='/Users/juanromero/Documents/python_3_Files/test'
num_maps=10

num_patients_per_trial_arm=10
num_baseline_months=2
num_testing_months=3
max_min_req_base_sz_count=4

placebo_mu=0
placebo_sigma=0.05
drug_mu=0.2
drug_sigma=0.05


for min_req_base_sz_count in $(seq 0 1 $max_min_req_base_sz_count);
do
    for map_num in $(seq 1 1 $num_maps);
    do
        for monthly_mean in $(seq $start_monthly_mean $step_monthly_mean $stop_monthly_mean);
        do
            for monthly_std_dev in $(seq $start_monthly_std_dev $step_monthly_std_dev $stop_monthly_std_dev);
            do
                inputs[0]=$monthly_mean
                inputs[1]=$monthly_std_dev
                inputs[2]=$num_trials
                inputs[3]=$directory
                inputs[4]=$map_num
                inputs[5]=$num_patients_per_trial_arm
                inputs[6]=$num_baseline_months
                inputs[7]=$num_testing_months
                inputs[8]=$min_req_base_sz_count
                inputs[9]=$placebo_mu
                inputs[10]=$placebo_sigma
                inputs[11]=$drug_mu
                inputs[12]=$drug_sigma

                # put a wrapper script here later
                python generate_map_point.py ${inputs[@]}
            done
        done
    done
done