#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-04:50
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

start=$SECONDS

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

num_training_files_per_block=${19}
num_testing_files_per_block=${20}
data_storage_folder_name=${17}
block_num=${18}

count=0
while ((count < 20))
do

    inputs[16]=$block_num

    inputs[17]="${data_storage_folder_name}/training_data"

    for ((file_index=1; file_index<=$num_training_files_per_block; file_index=file_index+1))
    do
        inputs[18]=$file_index

        sbatch keras_data_generation_wrapper.sh ${inputs[@]}

    done

    inputs[17]="${data_storage_folder_name}/testing_data"

    for ((file_index=1; file_index<=$num_testing_files_per_block; file_index=file_index+1))
    do
        inputs[18]=$file_index

        sbatch keras_data_generation_wrapper.sh ${inputs[@]}

    done

    block_num=$((block_num + 1))
    count=$((count + 1))

done

next_inputs[0]=$1
next_inputs[1]=$2
next_inputs[2]=$3
next_inputs[3]=$4
next_inputs[4]=$5
next_inputs[5]=$6
next_inputs[6]=$7
next_inputs[7]=$8
next_inputs[8]=$9
next_inputs[9]=${10}
next_inputs[10]=${11}
next_inputs[11]=${12}
next_inputs[12]=${13}
next_inputs[13]=${14}
next_inputs[14]=${15}
next_inputs[15]=${16}
next_inputs[16]=$data_storage_folder_name
next_inputs[17]=$block_num
next_inputs[18]=$num_training_files_per_block
next_inputs[19]=$num_testing_files_per_block

done=0
while (( done == 0 ))
do
    sleep 1m
    end=$SECONDS
    time=$((end - start))
    if ((time > 16200)); then
        done=1
        sbatch submit_one_block.sh ${next_inputs[@]}
    fi
done

