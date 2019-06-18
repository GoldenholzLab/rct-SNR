
# The information needed for the x-axis of the endpoint statistic maps
inputs[0]=${1}
inputs[1]=${2}
inputs[2]=${3}

# The information needed for the y-axis of the endpoint statistic maps
inputs[3]=${4}
inputs[4]=${5}
inputs[5]=${6}

# The parameters for estimating the value at each point on the endpoint statistic maps
inputs[6]=${7}
inputs[7]=${8}
inputs[8]=${9}
inputs[9]=${10}
inputs[10]=${11}

# The parameters for generating the placebo and drug effects
inputs[11]=${12}
inputs[12]=${13}
inputs[13]=${14}
inputs[14]=${15}

# The parameters needed for generating the histograms of the model 1 and model 2 patients
inputs[15]=${16}
inputs[16]=${17}

# Set the location of the directory containing the folder in which all the intermediate JSON files for this specific map will be stored
inputs[17]=${18}

# Set the name of the folder in which all the intermediate JSON files for this specific map will be stored
inputs[18]=${19}

python generate_data.py ${inputs[@]}
