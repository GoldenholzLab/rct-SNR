#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-08:30
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

#srun -c 1 python emp_and_map_based_analysis.py ${inputs[@]}
python -u emp_and_map_based_analysis.py ${inputs[@]}
