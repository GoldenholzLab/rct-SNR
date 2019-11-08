#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:05
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

num_estims=100

for ((x=1; x<=$num_estims; x=x+1))
do
    if [ ! -f RR50_smart/$x.txt ]
    then
        sbatch RR50_or_MPC_cost_wrapper.sh RR50 smart $x
    fi
    : '
    if [ ! -f MPC_smart/$x.txt ]
    then
        sbatch RR50_or_MPC_cost_wrapper.sh MPC  smart $x
    fi
    '
    #sbatch TTP_cost_wrapper.sh              smart $x
    if [ ! -f RR50_dumb/$x.txt ]
    then
        sbatch RR50_or_MPC_cost_wrapper.sh RR50 dumb  $x
    fi
    : '
    if [ ! -f MPC_dumb/$x.txt ]
    then
        sbatch RR50_or_MPC_cost_wrapper.sh MPC  dumb  $x
    fi
    '
    #sbatch TTP_cost_wrapper.sh              dumb  $x
done

: '
num_estims=10

for ((x=1; x<=$num_estims; x=x+1))
do
    bash RR50_or_MPC_cost_wrapper.sh RR50 smart $x
    bash RR50_or_MPC_cost_wrapper.sh MPC  smart $x
    bash TTP_cost_wrapper.sh              smart $x
    bash RR50_or_MPC_cost_wrapper.sh RR50 dumb  $x
    bash RR50_or_MPC_cost_wrapper.sh MPC  dumb  $x
    bash TTP_cost_wrapper.sh              dumb  $x
done
'