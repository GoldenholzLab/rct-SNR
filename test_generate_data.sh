
# The information needed for the x-axis of the expected placebo response maps
inputs[0]=0
inputs[1]=16
inputs[2]=1

# The information needed for the y-axis of the expected placebo response maps
inputs[3]=0
inputs[4]=16
inputs[5]=1

# The parameters for estimating the placebo response at each point on the expected placebo response maps
inputs[6]=2
inputs[7]=3
inputs[8]=4
inputs[9]=153
inputs[10]=10

# The parameters for generating the placebo and drug effects
inputs[11]=0
inputs[12]=0.05
inputs[13]=0.2
inputs[14]=0.05

# The parameters needed for generating the histograms of the model 1 and model 2 patients
inputs[15]=10000
inputs[16]=24

# The names of the files where the data/metadata will be stored
inputs[17]='RR50_stat_power_map'
inputs[18]='RR50_stat_power_map_metadata'
inputs[19]='MPC_stat_power_map'
inputs[20]='MPC_stat_power_map_metadata'
inputs[21]='TTP_stat_power_map'
inputs[22]='TTP_stat_power_map_metadata'
inputs[23]='H_model_1_hist'
inputs[24]='H_model_1_hist_metadata'
inputs[25]='H_model_2_hist'
inputs[26]='H_model_2_hist_metadata'

# The name of text file which will contain the placebo responses for NV model 1 and NV model 2
inputs[27]='NV_model_statistical_power'

python generate_data.py ${inputs[@]}
