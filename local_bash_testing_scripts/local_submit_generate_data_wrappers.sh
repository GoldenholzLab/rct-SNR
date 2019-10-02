
inputs[0]=$1
inputs[1]=$2
inputs[2]=$3
inputs[3]=$4

inputs[4]=$5
inputs[5]=$6
inputs[6]=$7
inputs[7]=$8

inputs[8]=$9
inputs[9]=${10}
inputs[10]=${11}
inputs[11]=${12}

inputs[12]=${13}
inputs[13]=${14}
inputs[14]=${15}

for ((compute_iter=1; compute_iter<${16}+1; compute_iter=compute_iter+1))
do
    inputs[15]=$compute_iter

    #sbatch generate_data_wrapper.sh ${inputs[@]}
    bash local_generate_data_wrapper.sh ${inputs[@]}

done

