#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-01:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

inputs[0]=0
inputs[1]=15
inputs[2]=0.1
inputs[3]=0
inputs[4]=15
inputs[5]=0.1
inputs[6]=150
inputs[7]=30
inputs[8]=28
inputs[9]=2
inputs[10]=3
inputs[11]=10000
inputs[12]=24
inputs[13]='model_1_and_model_2_expected_endpoints'
inputs[14]=3
inputs[15]='True'

srun -c 1 python endpoint_expectations.py ${inputs[@]}