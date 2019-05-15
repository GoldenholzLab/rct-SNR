
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

# The names of the files where the data/metadata will be stored
inputs[11]='expected_RR50_map'
inputs[12]='expected_RR50_map_metadata'
inputs[13]='expected_MPC_map'
inputs[14]='expected_MPC_map_metadata'

python generate_data.py ${inputs[@]}
