
inputs[0]=0
inputs[1]=16
inputs[2]=1
inputs[3]=0
inputs[4]=16
inputs[5]=1
inputs[6]=2
inputs[7]=3
inputs[8]=4
inputs[9]=153
inputs[10]=10
inputs[7]='expected_RR50_map'
inputs[8]='expected_RR50_map_metadata'
inputs[9]='expected_MPC_map'
inputs[10]='expected_MPC_map_metadata'

python generate_data.py ${inputs[@]}
