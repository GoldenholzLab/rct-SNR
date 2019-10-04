#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=500M
#SBATCH -t 0-00:15
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
inputs[15]=${18}

inputs_two[0]=$1
inputs_two[1]=$2
inputs_two[2]=$3
inputs_two[3]=$4
inputs_two[4]=${15}
inputs_two[5]=${17}
inputs_two[6]=${18}

inputs_three[0]=$1
inputs_three[1]=$2
inputs_three[2]=$3
inputs_three[3]=$4
inputs_three[4]=$5
inputs_three[5]=$6
inputs_three[6]=$7
inputs_three[7]=$8
inputs_three[8]=$9
inputs_three[9]=${10}
inputs_three[10]=${11}
inputs_three[11]=${12}
inputs_three[12]=${13}
inputs_three[13]=${14}
inputs_three[14]=${16}
inputs_three[15]=${19}

inputs_four[0]=$1
inputs_four[1]=$2
inputs_four[2]=$3
inputs_four[3]=$4
inputs_four[4]=${16}
inputs_four[5]=${17}
inputs_four[6]=${19}


sbatch submit_generate_data_wrappers.sh ${inputs[@]}
#bash local_submit_generate_data_wrappers.sh ${inputs[@]}

all_training_files_exist='False'
while [ "$all_training_files_exist" == "False" ]
do
    sleep 1
    if [ -d ${15} ]
    then
        echo `pwd`
        echo `ls "${15}"`
        echo ' ' 
        echo `ls "${15}/RR50_emp_stat_powers_"*`
        echo ' ' 
        echo `ls "${15}/RR50_emp_stat_powers_*"`
        echo ' ' 
        echo `ls "${15}/RR50_emp_stat_powers_"*".json"`
        echo ' ' 
        echo `ls "${15}/RR50_emp_stat_powers_*.json"`
        echo ' '
        echo ' ' 
        x1=`ls -1 "${15}/RR50_emp_stat_powers_*.json" | wc -l`
        x2=`ls -1 "${15}/theo_placebo_arm_hists_*.json" | wc -l`
        x3=`ls -1 "${15}/theo_drug_arm_hists_*.json" | wc -l`
        if [[ $x1 == $num_compute_training_iters && $x2 == $num_compute_training_iters && $x3 == $num_compute_training_iters ]]
        then
            all_training_files_exist='True'
            sbatch train_model_wrapper.sh ${inputs_two[@]}
            #bash local_train_model_wrapper.sh ${inputs_two[@]}
        fi
    fi
done

sbatch submit_generate_data_wrappers.sh ${inputs_three[@]}
#bash local_submit_generate_data_wrappers.sh ${inputs_three[@]}

all_testing_files_exist='False'
while [ "$all_testing_files_exist" == "False" ]
do
    sleep 1
    if [ -d ${16} ∂]
    then
        x1=`ls -1 "${16}/RR50_emp_stat_powers_*.json" | wc -l`
        x2=`ls -1 "${16}/theo_placebo_arm_hists_*.json" | wc -l`
        x3=`ls -1 "${16}/theo_drug_arm_hists_*.json" | wc -l`
        if [[ $x1 == $num_compute_testing_iters && $x2 == $num_compute_testing_iters && $x3 == $num_compute_testing_iters && -f "RR50_stat_power_model_trained.h5" ]]
        then
            all_testing_files_exist='True'
            sbatch test_model_wrapper.sh ${inputs_four[@]}
            #bash local_test_model_wrapper.sh ${inputs_four[@]}
        fi
    fi
done

