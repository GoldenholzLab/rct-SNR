#!/usr/bin/bash

#SBATCH -p short
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

a=1
while [ "$a" -eq 1 ]
do
    if [ -f "yodeling.txt" ]
    then
        a=0
    fi
    sleep 1
done

echo ${inputs[@]}
sbatch train_model_wrapper.sh ${inputs[@]}