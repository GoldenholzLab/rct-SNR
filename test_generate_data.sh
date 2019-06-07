
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

# The parameters needed for generating the histograms of the model 1 and model 2 patients
inputs[11]=10000
inputs[12]=24

# The names of the files where the data/metadata will be stored
inputs[13]='expected_RR50_map'
inputs[14]='expected_RR50_map_metadata'
inputs[15]='expected_MPC_map'
inputs[16]='expected_MPC_map_metadata'
inputs[17]='H_model_1_hist'
inputs[18]='H_model_1_hist_metadata'
inputs[19]='H_model_2_hist'
inputs[20]='H_model_2_hist_metadata'

# The name of text file which will contain the placebo responses for NV model 1 and NV model 2
inputs[21]='NV_model_placebo_response'

python generate_data.py ${inputs[@]}
