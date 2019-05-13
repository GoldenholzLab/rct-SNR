
inputs[0]=0
inputs[1]=16
inputs[2]=1
inputs[3]=0
inputs[4]=16
inputs[5]=1
inputs[6]=2
inputs[7]=2
inputs[8]=153
inputs[9]=100
inputs[10]=2
inputs[11]=3
inputs[12]=4
inputs[13]=10000
inputs[14]=24
inputs[15]=28
inputs[16]=1.3
inputs[17]=0.5
inputs[18]=0.1
inputs[19]=1
inputs[20]='expected_RR50'
inputs[21]='expected_MPC'
inputs[22]='expected_RR50_with_power_law_curves'
inputs[23]='expected_MPC_with_power_law_curves'
inputs[24]='Model_1_2D_histogram'
inputs[25]='Model_2_2D_histogram'
inputs[26]='Model_1_and_Model_2_expected_endpoints'

python endpoint_expectations.py ${inputs[@]}