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
inputs[1]=16
inputs[2]=1
inputs[3]=0
inputs[4]=16
inputs[5]=1
inputs[6]=2
inputs[7]=2
inputs[8]=1000
inputs[9]=24
inputs[10]=95
inputs[11]=2
inputs[12]=1000
inputs[13]="Gaussianity_map"
inputs[14]="Gaussianity_map_with_models"

python lemma_check_v2.py ${inputs[@]}