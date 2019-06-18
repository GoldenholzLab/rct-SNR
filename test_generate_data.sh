
# The information needed for the x-axis of the endpoint statistic maps
inputs[0]=0
inputs[1]=16
inputs[2]=1

# The information needed for the y-axis of the endpoint statistic maps
inputs[3]=0
inputs[4]=16
inputs[5]=1

# The parameters for estimating the value at each point on the endpoint statistic maps
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

# set the name of text file which will contain the placebo responses for NV model 1 and NV model 2
inputs[17]='NV_model_endpoint_statistics'

# give the location of the folder in which all the intermediate JSON files for this specific map will be stored
inputs[18]='/Users/juanromero/Documents/GitHub/rct-SNR'

python generate_data.py ${inputs[@]}
