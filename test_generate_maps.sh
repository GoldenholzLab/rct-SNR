
# The information needed for the monthly seizure frequency axis of the endpoint statistic maps
# These parameters need to be passed off as metadata to plotting scripts
mu_start=0
mu_stop=12
mu_step=1

# The information needed for the monthly seizure standard deviation axis of the endpoint statistic maps
# These parameters need to be passed off as metadata to plotting scripts
sigma_start=0
sigma_stop=12
sigma_step=1

# The parameters for estimating the value at each point on the endpoint statistic maps
num_baseline_months=2
num_testing_months=3
max_num_req_baseline_sz=2
num_patients_per_arm=20
num_trials_per_map_point=2

# The parameters for generating the placebo and drug effects
placebo_mean=0
placebo_sigma=0.05
drug_mean=0.2
drug_sigma=0.05

# set the number of maps to average over
num_maps=3

# The location of the directory containing the folder in which all the intermediate JSON files for this specific map will be stored
directory='/Users/juanromero/Documents/Python_3_Files/test'

# create a new meta-data text file
touch $directory/meta_data.txt

# write information about the monthly seizure count mean axis into the text file
echo $mu_start >> $directory/meta_data.txt
echo $mu_stop >> $directory/meta_data.txt
echo $mu_step >> $directory/meta_data.txt

# write information about the monthly seizure count standard deviation axis into the text file
echo $sigma_start >> $directory/meta_data.txt
echo $sigma_stop >> $directory/meta_data.txt
echo $sigma_step >> $directory/meta_data.txt

# write information about the maximum of the minimum required number of seizures in the baseline period
echo $max_num_req_baseline_sz >> $directory/meta_data.txt

# write information about the number of maps to average over into the text file
echo $num_maps >> $directory/meta_data.txt


# loop over all the minimum required numbers of seizures in the baseline period, from 0 all the way up to the maximum
for ((num_req_baseline_sz=0; num_req_baseline_sz<=$max_num_req_baseline_sz; num_req_baseline_sz=num_req_baseline_sz+1));
    do
        # loop over all the undersampled map folders to take the average over
        for ((folder_num=1; folder_num<=$num_maps; folder_num=folder_num+1));
            do
                # store the information needed for the x-axis of the endpoint statistic maps
                inputs[0]=$mu_start
                inputs[1]=$mu_stop
                inputs[2]=$mu_step

                # store the information needed for the y-axis of the endpoint statistic maps
                inputs[3]=$sigma_start
                inputs[4]=$mu_stop
                inputs[5]=$mu_step

                # store the parameters for estimating the value at each point on the endpoint statistic maps
                inputs[6]=$num_baseline_months
                inputs[7]=$num_testing_months
                inputs[8]=$num_req_baseline_sz
                inputs[9]=$num_patients_per_arm
                inputs[10]=$num_trials_per_map_point

                # store the parameters for generating the placebo and drug effects
                inputs[11]=$placebo_mean
                inputs[12]=$placebo_sigma
                inputs[13]=$drug_mean
                inputs[14]=$drug_sigma
                
                # store the location of the directory containing the folder in which all the intermediate JSON files for this specific map will be stored
                inputs[15]=$directory

                # store the name of the folder in which all the intermediate JSON files for this specific map will be stored
                inputs[16]=$folder_num

                python generate_maps.py ${inputs[@]}
        done
done
