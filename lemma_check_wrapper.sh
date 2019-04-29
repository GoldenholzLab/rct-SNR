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
inputs[6]=1000
inputs[7]=24
inputs[8]=28
inputs[9]=95
inputs[10]=1000
inputs[11]=2
inputs[12]=10
inputs[13]="lemma_check_map_with_models"
inputs[14]="lemma_check_map"
inputs[15]="prob_successes"

srun -c 1 python lemma_check.py ${inputs[@]}
