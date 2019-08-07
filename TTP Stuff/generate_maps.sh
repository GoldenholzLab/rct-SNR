
monthly_mean_min=4
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=8

folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'

inputs[1]=$monthly_mean_min
inputs[2]=$monthly_mean_max
inputs[3]=$monthly_std_dev_min
inputs[4]=$monthly_std_dev_max
inputs[5]=$folder

python generate_maps.py ${inputs[@]}
