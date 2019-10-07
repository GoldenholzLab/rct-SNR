#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=500M
#SBATCH -t 0-00:10
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

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
inputs[15]=${16}

for ((compute_iter=1; compute_iter<${17}+1; compute_iter=compute_iter+1))
do
    inputs[16]=$compute_iter

    sbatch generate_data_wrapper.sh ${inputs[@]}
    #bash local_generate_data_wrapper.sh ${inputs[@]}

done

