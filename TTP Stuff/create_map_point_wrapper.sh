#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-02:15
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
module load R/3.5.1
source activate main_env

inputs[1]=$1
inputs[2]=$2
inputs[3]=$3
inputs[4]=$4
inputs[5]=$5
inputs[6]=$6
inputs[7]=$7
inputs[8]=$8
inputs[9]=$9
inputs[10]=${10}
inputs[11]=${11}
inputs[12]=${12}

python -u create_map_point.py ${inputs[@]}
