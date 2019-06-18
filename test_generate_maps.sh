
# The information needed for the monthly seizure frequency axis of the endpoint statistic maps
# These parameters need to be passed off as metadata to plotting scripts
mu_start=0
mu_stop=16
mu_step=1

# The information needed for the monthly seizure standard deviation axis of the endpoint statistic maps
# These parameters need to be passed off as metadata to plotting scripts
sigma_start=0
sigma_stop=16
sigma_step=1

# The parameters for estimating the value at each point on the endpoint statistic maps
num_baseline_months=2
num_testing_months=3
num_patients_per_arm=20
num_trials_per_map_point=2

# The parameters for generating the placebo and drug effects
placebo_mean=0
placebo_sigma=0.05
drug_mean=0.2
drug_sigma=0.05

# The parameters needed for generating the histograms of the model 1 and model 2 patients
num_patients_per_NV_model=10000
num_months_per_NV_model_patient=24

# The location of the directory containing the folder in which all the intermediate JSON files for this specific map will be stored
directory='/Users/juanromero/Documents/GitHub/rct-SNR'

for ((num_req_baseline_sz=0; num_req_baseline_sz<5; num_req_baseline_sz=num_req_baseline_sz+1));
    do
        for ((folder=1; folder<6; folder=folder+1));
            do
                inputs[0]=$mu_start
                inputs[1]=$mu_stop
                inputs[2]=$mu_step
                inputs[3]=$sigma_start
                inputs[4]=$mu_stop
                inputs[5]=$mu_step
                inputs[6]=$num_baseline_months
                inputs[7]=$num_testing_months
                inputs[8]=$num_req_baseline_sz
                inputs[9]=$num_patients_per_arm
                inputs[10]=$num_trials_per_map_point
                inputs[11]=$placebo_mean
                inputs[12]=$placebo_sigma
                inputs[13]=$drug_mean
                inputs[14]=$drug_sigma
                inputs[15]=$num_patients_per_NV_model
                inputs[16]=$num_months_per_NV_model_patient
                inputs[17]=$directory
                inputs[18]=$folder

                bash test_generate_map.sh ${inputs[@]}
        done
done
