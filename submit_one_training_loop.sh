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
inputs[15]=${20}
inputs[16]=${18}

inputs_two[0]=$1
inputs_two[1]=$2
inputs_two[2]=$3
inputs_two[3]=$4
inputs_two[4]=${15}
inputs_two[5]=${17}
inputs_two[6]=${18}
inputs_two[7]=${20}

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
inputs_three[15]=${20}
inputs_three[16]=${19}

inputs_four[0]=$1
inputs_four[1]=$2
inputs_four[2]=$3
inputs_four[3]=$4
inputs_four[4]=${16}
inputs_four[5]=${17}
inputs_four[6]=${19}
inputs_four[7]=${20}

: '
inputs_five[0]=$1
inputs_five[1]=$2
inputs_five[2]=$3
inputs_five[3]=$4
inputs_five[4]=$5
inputs_five[5]=$6
inputs_five[6]=$7
inputs_five[7]=$8
inputs_five[8]=$9
inputs_five[9]=${10}
inputs_five[10]=${11}
inputs_five[11]=${12}
inputs_five[12]=${13}
inputs_five[13]=${14}
inputs_five[14]=${15}
inputs_five[15]=${16}
inputs_five[16]=${17}
inputs_five[17]=${18}
inputs_five[18]=${19}
inputs_five[19]=$((${20} + 1))
'


sbatch submit_generate_data_wrappers.sh ${inputs[@]}

all_training_files_exist='False'
while [ "$all_training_files_exist" == "False" ]
do
    sleep 15
    if [ -d "${15}_${20}" ]
    then
        x1=`ls -1 "${15}_${20}/RR50_emp_stat_powers_"* | wc -l`
        x2=`ls -1 "${15}_${20}/theo_placebo_arm_hists_"* | wc -l`
        x3=`ls -1 "${15}_${20}/theo_drug_arm_hists_"* | wc -l`
        if [ $x1 == ${18} ]
        then
            echo "$x1 1"
        fi
        if [ $x2 == ${18} ]
        then
            echo "$x2 2"
        fi
        if [ $x3 == ${18} ]
        then
            echo "$x3 3"
        fi
        if [ $x1 == ${18} ] && [ $x2 == ${18} ] && [ $x3 == ${18} ]
        then
            echo 'reached'
            all_training_files_exist='True'
            sbatch train_model_wrapper.sh ${inputs_two[@]}
        else
            echo 'not reached'
        fi
    fi
done

sbatch submit_generate_data_wrappers.sh ${inputs_three[@]}

: '
while [ ! -f "${17}_${20}_trained.h5" ]
do
    sleep 15
done

sbatch "$0" "${inputs_five[@]}"
'

all_testing_files_exist='False'
while [ "$all_testing_files_exist" == "False" ]
do
    sleep 15
    if [ -d "${16}_${20}" ]
    then
        x1=`ls -1 "${16}_${20}/RR50_emp_stat_powers_"* | wc -l`
        x2=`ls -1 "${16}_${20}/theo_placebo_arm_hists_"* | wc -l`
        x3=`ls -1 "${16}_${20}/theo_drug_arm_hists_"* | wc -l`
        if [ $x1 == ${19} ]
        then
            echo "$x1 4"
        fi
        if [ $x2 == ${19}  ]
        then
            echo "$x2 5"
        fi
        if [ $x3 == ${19}  ]
        then
            echo "$x3 6"
        fi
        if [ -f "${17}_${20}_trained.h5" ]
        then
            echo 'trained model exists'
        fi
        if [ $x1 = ${19} ] && [ $x2 = ${19} ] && [ $x3 = ${19} ] && [ -f "${17}_${20}_trained.h5" ]
        then
            echo 'reached 2'
            all_testing_files_exist='True'
            sbatch test_model_wrapper.sh ${inputs_four[@]}
        else
            echo 'not reached 2'
        fi
    fi
done

